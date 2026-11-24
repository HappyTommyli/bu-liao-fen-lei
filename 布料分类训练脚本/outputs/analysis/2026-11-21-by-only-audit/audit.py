"""Read-only audit using the production loader with an in-memory By-only quality mask."""
import sys,json,csv,hashlib,collections,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from cloth_trainer.data import discover_report,load_record,Unassessable
from cloth_trainer.config import DataConfig
OUT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'xz-control/data/布料采集数据新'
CONFIGS={'fixed_51_140':DataConfig(out_low=51,out_high=140,back_low=30,back_high=120,contact_rule='fixed-x-v1'), 'relative_035':DataConfig(out_low=51,out_high=140,back_low=30,back_high=120,contact_rule='relative-035-v1')}
REAL_READER=csv.DictReader

def number(v):
 try:return float(v)
 except (ValueError,TypeError):return float('nan')

def adjust(rows):
 y=np.array([number(r.get('By_uT',r.get('dBy_uT'))) for r in rows]);finite=np.isfinite(y);spike=np.zeros(len(rows),bool)
 if len(y)>2:spike[1:-1]=(abs(y[1:-1]-y[:-2])>=1000)&(abs(y[1:-1]-y[2:])>=1000)&(abs(y[2:]-y[:-2])<=250)
 changed=[];restored=0;original_invalid=0
 for i,row in enumerate(rows):
  r=dict(row);orig=str(r.get('sample_valid','1')).strip().lower() not in ('0','false','no');original_invalid+=int(not orig)
  if not orig and r.get('invalid_reason')=='isolated_spike' and finite[i] and not spike[i]:
   r['sample_valid']='1';r['invalid_reason']='';restored+=1
  if not finite[i] or spike[i]:r['sample_valid']='0';r['invalid_reason']='by_nonfinite' if not finite[i] else 'by_isolated_spike'
  changed.append(r)
 return changed,dict(original_invalid=original_invalid,restored_other_axis_spike=restored,by_isolated_spike=int(spike.sum()),by_nonfinite=int((~finite).sum()))

def check(path,cfg,rows,fields):
 class CachedReader:
  def __init__(self,*args,**kwargs):self.fieldnames=fields
  def __iter__(self):return iter(rows)
 try:
  with patch('cloth_trainer.data.csv.DictReader',CachedReader):r=load_record(path,cfg,signals=('by',))
  return dict(status='loaded',quality=r.quality,notes=r.quality_notes,warnings=r.warnings,diagnostics=r.diagnostics)
 except (ValueError,OSError,KeyError) as e:return dict(status='unassessable' if isinstance(e,Unassessable) else 'error',error=str(e))

def main():
 paths,ignored,errors=discover_report([SOURCE]);records=[];start=time.time()
 for idx,p in enumerate(paths,1):
  raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();mp=p.parent/'metadata.json';mb=mp.read_bytes() if mp.exists() else None;m=json.loads(mb) if mb else {}
  with p.open(encoding='utf-8-sig',newline='') as f:reader=REAL_READER(f);fields=reader.fieldnames;rows=list(reader)
  new,detail=adjust(rows)
  record=dict(path=str(p),name=p.stem,label=m.get('label',p.stem.rsplit('_',1)[0]),created_at=m.get('created_at'),outcome=m.get('outcome'),rows=len(rows),csv_sha256=sha,metadata_sha256=hashlib.sha256(mb).hexdigest() if mb else None,quality_mask=detail,checks={})
  for rule,cfg in CONFIGS.items():record['checks'][rule]=dict(original=check(p,cfg,rows,fields),by_only=check(p,cfg,new,fields))
  record['unchanged_after_check']=(p.read_bytes()==raw and (mp.read_bytes() if mp.exists() else None)==mb)
  records.append(record)
  if idx%50==0:print('progress',idx,'/',len(paths),'seconds',round(time.time()-start),flush=True)
  (OUT/'audit.partial.json').write_text(json.dumps(dict(scanned=idx,total=len(paths),records=records),ensure_ascii=False)) if idx%100==0 else None
 result=dict(source=str(SOURCE),configs={k:vars(v) for k,v in CONFIGS.items()},ignored=ignored,discovery_errors=errors,records=records,elapsed_seconds=time.time()-start)
 (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 for rule in CONFIGS:
  for mode in ('original','by_only'):
   print(rule,mode,dict(collections.Counter((r['checks'][rule][mode].get('quality') if r['checks'][rule][mode]['status']=='loaded' else r['checks'][rule][mode]['status']) for r in records)))
 print('DONE',len(records),'unchanged',all(r['unchanged_after_check'] for r in records),flush=True)
if __name__=='__main__':main()
