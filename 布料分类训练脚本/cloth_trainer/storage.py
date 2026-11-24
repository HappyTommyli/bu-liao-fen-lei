from pathlib import Path
from datetime import datetime
import json
import uuid
import zipfile
import shutil

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def new_task(root,prefix='CERA'):
    p=Path(root)/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);p.mkdir(parents=True);return p

def list_tasks(root):
    found=[]
    for p in Path(root).glob('*/metadata.json'):
        try:
            m=read_json(p);m['directory']=str(p.parent.resolve());found.append(m)
        except (ValueError,OSError):pass
    return sorted(found,key=lambda m:m.get('created_at',''),reverse=True)

def export_task(task,destination):
    task=Path(task).resolve();destination=Path(destination).resolve()
    if task in destination.parents:raise ValueError('请将成果包保存到任务文件夹之外')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(task.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':z.write(p,p.relative_to(task))

def import_task(package,root):
    target=new_task(root)
    try:
        with zipfile.ZipFile(package) as z:
            for info in z.infolist():
                if not (target/info.filename).resolve().is_relative_to(target.resolve()):raise ValueError('成果包包含越界路径')
            z.extractall(target)
        meta=read_json(target/'metadata.json')
        if meta.get('format') not in ('cera-net-v1','cloth-classification-v2','cloth-classification-v3','cloth-classification-v4') or meta.get('status')!='completed':raise ValueError('需要本工具已完成的成果包')
        for name in meta['models']:
            if not (target/name/'model.pt').resolve().is_relative_to(target.resolve()) or not (target/name/'model.pt').is_file():raise ValueError('模型文件不完整')
        from .engine import load_bundle
        if meta['format']=='cloth-classification-v4':
            from .modern_engine import frozen
            from .data import file_hash
            frozen(target)
            for name,r in meta['models'].items():
                if file_hash(target/name/'model.pt')!=r['weight_sha256']:raise ValueError('开发集权重校验失败')
            if meta.get('final_model'):load_bundle(target/'final/model.pt',allow_quick=True)
        else:
            for name in meta['models']:load_bundle(target/name/'model.pt',allow_quick=True)
        meta.update(id=target.name,name=meta['name']+'（导入）');write_json(target/'metadata.json',meta)
        return target
    except Exception:
        shutil.rmtree(target);raise
