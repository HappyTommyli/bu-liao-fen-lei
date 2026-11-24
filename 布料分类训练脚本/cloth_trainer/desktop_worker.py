"""Isolated stdlib-only JSON parsing for the desktop (no Qt/PyTorch import)."""
import json
import sys
import os
from pathlib import Path


def compact_metadata(m):
    # Raw audit payloads stay in the original artifact, not every combo/list view.
    result={k:v for k,v in m.items() if k not in ('screening','admission')}
    result['sources']=[{k:v for k,v in s.items() if k in ('path','sha256','metadata_sha256','label','group')} for s in m.get('sources',[])]
    return result


def catalog_entry(m,directory):
    keep=('format','name','status','created_at','purpose','validation_kind','records','labels','config','dataset_id','split_plan','study','study_description')
    out={k:m[k] for k in keep if k in m};out['directory']=str(Path(directory).resolve())
    fields=('architecture','signal','readout','status','candidate','configuration_id','fold','seed','phase','parameters','best_epoch','weight_sha256')
    out['models']={k:{f:r[f] for f in fields if f in r} for k,r in m.get('models',{}).items()}
    final=m.get('final_model');out['final_model']={k:final[k] for k in fields if k in final} if final else None
    out['sources']=[{k:s[k] for k in ('path','sha256','metadata_sha256','label','group') if k in s} for s in m.get('sources',[])]
    return out


def execute(kind,payload):
    if kind=='finish-task':
        path=Path(payload['path']);meta=json.loads(path.read_text(encoding='utf-8'))
        if meta.get('status')=='running':
            meta.update(status='cancelled' if payload.get('cancelled') else 'failed',error=payload.get('error') or '后台进程异常退出')
            tmp=path.with_name(path.name+'.finish.tmp');tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
        return compact_metadata(meta)
    if kind=='metadata':return compact_metadata(json.loads(Path(payload['path']).read_text(encoding='utf-8')))
    if kind=='catalog':
        rows=[]
        for path in Path(payload['root']).glob('*/metadata.json'):
            try:
                stat=path.stat();cache=path.with_name('.desktop-summary.json');signature=[stat.st_mtime_ns,stat.st_size]
                saved=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
                if saved.get('signature')==signature:row=dict(saved['entry'],directory=str(path.parent.resolve()))
                else:
                    row=catalog_entry(json.loads(path.read_text(encoding='utf-8')),path.parent)
                    try:
                        tmp=cache.with_name(cache.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(dict(signature=signature,entry=row),ensure_ascii=False));tmp.replace(cache)
                    except OSError:pass
                rows.append(row)
            except (OSError,ValueError,KeyError):continue
        return sorted(rows,key=lambda m:m.get('created_at',''),reverse=True)
    raise ValueError('未知界面后台操作')


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));destination=Path(sys.argv[2])
    try:result=dict(ok=True,value=execute(request['kind'],request['payload']))
    except Exception as e:result=dict(ok=False,error=str(e))
    destination.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
