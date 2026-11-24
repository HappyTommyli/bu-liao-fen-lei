"""Fabric-labelled recordings, durable counters, and legacy record renaming."""
import fcntl
import json
import os
import re
import uuid
import tempfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

DEFAULT_LABEL='未命名布料'
DEFAULT_SETTINGS=Path(os.environ.get('XZ_WORKBENCH_SETTINGS',Path(__file__).resolve().parents[1]/'preview-runtime'/'settings.json'))


def read_json(path,default=None):
    try:
        result=json.loads(Path(path).read_text(encoding='utf-8'))
        return result if isinstance(result,dict) else ({} if default is None else default)
    except (OSError,ValueError): return {} if default is None else default


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive random temp creation prevents following a pre-existing symlink.
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix=path.name+'.',suffix='.tmp',delete=False) as out:
            temp=Path(out.name)
            json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False)
            out.flush(); os.fsync(out.fileno())
        temp.replace(path)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)



def valid_label(value):
    if not isinstance(value,str): raise ValueError('布料名称必须是文字')
    value=value.strip()
    if not value or len(value)>80 or value.startswith('.') or value.endswith('.') or re.search(r'[\\/<>:"|?*\x00-\x1f\x7f]',value):
        raise ValueError('名称请使用 1–80 个字符，不含路径符号、控制字符或首尾句点')
    return value


class Catalog:
    def __init__(self,root,settings_path=None):
        self.root=Path(root).absolute()
        self.settings_path=Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS

    def last_label(self):
        value=read_json(self.settings_path).get('last_label',DEFAULT_LABEL)
        try: return valid_label(value)
        except ValueError: return DEFAULT_LABEL

    def remember(self,label):
        from .preferences import update_settings
        update_settings(self.settings_path,{'last_label':valid_label(label)})

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'.workbench.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(lock,fcntl.LOCK_UN)

    def _next(self,label):
        # The real directory is authoritative; old high-water counters are ignored.
        occupied=set()
        pattern=re.compile(re.escape(label)+r'_(\d+)$')
        if self.root.exists():
            for path in self.root.iterdir():
                match=pattern.fullmatch(path.name)
                if match: occupied.add(int(match[1]))
        n=1
        while n in occupied: n+=1
        return n,read_json(self.root/'.workbench-index.json')

    def next_name(self,label):
        label=valid_label(label); n,_=self._next(label)
        return f'{label}_{n:03d}'

    def _reserve(self,label):
        n,counters=self._next(label)
        while True:
            name=f'{label}_{n:03d}'; path=self.root/name
            try: path.mkdir(); break
            except FileExistsError: n+=1
        counters[label]=n
        # Keep reservation if persistence fails; occupied folders are never overwritten.
        atomic_json(self.root/'.workbench-index.json',counters)
        return dict(path=path,label=label,run_name=name,sequence=n,csv_name=name+'.csv')

    def reserve(self,label):
        label=valid_label(label)
        with self.locked(): result=self._reserve(label)
        self.remember(label)
        return result

    def records(self):
        rows=[]
        if not self.root.exists(): return rows
        for path in self.root.iterdir():
            if not path.is_dir() or path.is_symlink() or path.name.startswith('.'): continue
            meta=read_json(path/'metadata.json')
            if not isinstance(meta,dict) or not meta: continue
            rows.append(dict(path=str(path),name=path.name,label=meta.get('label','未标注'),
                             created=meta.get('created_at',''),outcome=meta.get('outcome','interrupted'),
                             samples=meta.get('sample_count',0)))
        return sorted(rows,key=lambda r:(r['created'],r['name']),reverse=True)

    def rename(self,path,label,blocked=()):
        label=valid_label(label); path=Path(path).absolute()
        if path.is_symlink() or path.parent.resolve()!=self.root.resolve(): raise ValueError('请选择当前保存目录中的记录')
        if path.resolve() in {Path(p).resolve() for p in blocked}: raise ValueError('记录正在采集或后台保存，请稍后重命名')
        with self.locked():
            meta=read_json(path/'metadata.json')
            if meta.get('outcome') not in ('completed','aborted'): raise ValueError('仅能重命名已经结束保存的记录')
            csv_name=meta.get('files',{}).get('sensor','sensor.csv')
            if not isinstance(csv_name,str) or Path(csv_name).name!=csv_name: raise ValueError('记录 CSV 路径无效')
            source=path/csv_name
            if source.is_symlink() or not source.is_file(): raise ValueError('记录的 CSV 文件不存在或不是普通文件')
            allocation=self._reserve(label); target=allocation['path']; new_name=allocation['csv_name']
            if new_name!=csv_name and (path/new_name).exists(): raise ValueError('目标 CSV 文件已存在')
            original=dict(meta); moved=False; csv_moved=False
            try:
                # Replaces our empty reservation directory only; user data is never overwritten.
                path.rename(target); moved=True
                if csv_name!=new_name: (target/csv_name).rename(target/new_name); csv_moved=True
                meta.update(label=label,run_name=allocation['run_name'],sequence=allocation['sequence'])
                meta['files']=dict(meta.get('files',{}),sensor=new_name)
                atomic_json(target/'metadata.json',meta)
            except Exception:
                if moved:
                    if csv_moved: (target/new_name).rename(target/csv_name)
                    try: atomic_json(target/'metadata.json',original)
                    finally: target.rename(path)
                elif target.exists(): target.rmdir()
                raise
        self.remember(label)
        return target

    def _record(self,path,parent,blocked=()):
        path=Path(path).absolute()
        if path.is_symlink() or path.parent.resolve()!=parent.resolve() or path.name.startswith('.'):
            raise ValueError('请选择当前目录中的真实记录')
        if path.resolve() in {Path(p).resolve() for p in blocked}:
            raise ValueError('记录正在采集或后台保存，请稍后操作')
        if not path.is_dir() or (path/'metadata.json').is_symlink():
            raise ValueError('记录目录或元数据无效')
        meta=read_json(path/'metadata.json')
        if not meta or meta.get('outcome') not in ('completed','aborted','interrupted'):
            raise ValueError('记录元数据无效或仍在保存')
        csv_name=meta.get('files',{}).get('sensor','sensor.csv')
        if not isinstance(csv_name,str) or Path(csv_name).name!=csv_name:
            raise ValueError('记录 CSV 路径无效')
        if (path/csv_name).is_symlink() or not (path/csv_name).is_file():
            raise ValueError('记录 CSV 不是普通文件')
        return path,meta,csv_name

    def _trash_root(self):
        path=self.root/'.workbench-trash'
        if path.is_symlink(): raise ValueError('回收站不能是符号链接')
        return path

    def trash(self,paths,blocked=()):
        if not paths: raise ValueError('请先选择记录')
        with self.locked():
            records=[self._record(p,self.root,blocked) for p in dict.fromkeys(map(str,paths))]
            trash=self._trash_root(); trash.mkdir(exist_ok=True)
            moved=[]
            try:
                for path,meta,csv_name in records:
                    if (path/'.trash-info.json').exists(): raise ValueError('记录含有冲突的回收信息')
                    receipt=dict(original_name=path.name,deleted_at=datetime.now(timezone.utc).isoformat())
                    atomic_json(path/'.trash-info.json',receipt)
                    target=trash/uuid.uuid4().hex
                    try: path.rename(target)
                    except Exception:
                        (path/'.trash-info.json').unlink(); raise
                    moved.append((path,target))
            except Exception:
                for old,new in reversed(moved):
                    new.rename(old); (old/'.trash-info.json').unlink()
                raise
        return [new for old,new in moved]

    def trash_records(self):
        trash=self._trash_root(); rows=[]
        if not trash.exists(): return rows
        for path in trash.iterdir():
            if not path.is_dir() or path.is_symlink(): continue
            receipt=read_json(path/'.trash-info.json')
            if receipt:
                rows.append(dict(path=str(path),name=receipt.get('original_name',path.name),
                                 deleted=receipt.get('deleted_at','')))
        return sorted(rows,key=lambda r:r['deleted'],reverse=True)

    def restore(self,path,blocked=()):
        with self.locked():
            trash=self._trash_root(); path,meta,csv_name=self._record(path,trash,blocked)
            receipt=read_json(path/'.trash-info.json'); name=receipt.get('original_name','')
            if not isinstance(name,str) or not name or Path(name).name!=name or name.startswith('.'):
                raise ValueError('回收信息中的原记录名称无效')
            target=self.root/name; allocation=None
            if target.exists() or target.is_symlink():
                allocation=self._reserve(valid_label(meta.get('label',DEFAULT_LABEL))); target=allocation['path']
            new_csv=allocation['csv_name'] if allocation else csv_name
            original=dict(meta); moved=False; renamed=False
            try:
                if new_csv!=csv_name and (path/new_csv).exists(): raise ValueError('恢复 CSV 名称冲突')
                path.rename(target); moved=True
                if new_csv!=csv_name: (target/csv_name).rename(target/new_csv); renamed=True
                if allocation:
                    meta=dict(meta,label=allocation['label'],run_name=allocation['run_name'],sequence=allocation['sequence'])
                    meta['files']=dict(meta.get('files',{}),sensor=new_csv)
                    atomic_json(target/'metadata.json',meta)
                (target/'.trash-info.json').unlink()
            except Exception:
                if moved:
                    if renamed: (target/new_csv).rename(target/csv_name)
                    atomic_json(target/'metadata.json',original); target.rename(path)
                elif allocation: target.rmdir()
                raise
            return target
