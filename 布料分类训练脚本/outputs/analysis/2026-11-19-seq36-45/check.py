"""Reproduce import decisions for current captures 036-045; no mutations."""
import sys,json,csv,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import discover_report,load_record,file_hash
from cloth_trainer.config import DataConfig
DEST=Path(__file__).parent;SOURCE=ROOT.parent/'xz-control/data/布料采集数据'
paths,_,_=discover_report([SOURCE]);selected=[]
for p in paths:
    m=json.loads((p.parent/'metadata.json').read_text(encoding='utf-8-sig'))
    if m.get('label')not in ('三角形间隔','矩形间隔')and 36<=int(m.get('sequence')or 0)<=45:selected.append((p,m))
rows=[]
for i,(p,m)in enumerate(selected):
    row=dict(label=m['label'],record=p.stem,path=str(p),sequence=m['sequence'],created_at=m['created_at'],outcome=m.get('outcome'),
        saved_start=m['context'].get('capture_start_mm'),recipe=[(s['kind'],s.get('axis'),s.get('value'))for s in(m['context'].get('recipe')or{}).get('steps',[])if s.get('enabled',True)],
        sha256=file_hash(p),metadata_sha256=file_hash(p.parent/'metadata.json'))
    try:
        r=load_record(p,DataConfig(),signals=('by',));row.update(status=r.quality,eligible=r.audit['eligible'],reason='；'.join(r.audit['reasons']))
    except (ValueError,OSError,KeyError,TypeError)as exc:row.update(status='error',eligible=False,reason=str(exc))
    a=np.loadtxt(p,delimiter=',',skiprows=1,usecols=(7,8),dtype='U32',encoding='utf-8-sig',ndmin=2);ppmm=m['context']['config']['x_ppmm'];ranges={}
    for stage in ('step_003','step_004'):
        x=a[a[:,0]==stage,1].astype(float)/ppmm;ranges[stage]=[float(x.min()),float(x.max())]if len(x)else None
    row['ranges']=ranges;rows.append(row)
    if(i+1)%20==0 or i+1==len(selected):print('CHECK',i+1,len(selected),flush=True)
summary=[]
for label in sorted({r['label']for r in rows}):
    rr=[r for r in rows if r['label']==label];summary.append(dict(label=label,total=len(rr),usable=sum(r['eligible']for r in rr),reasons=dict(collections.Counter(r['reason']or'通过'for r in rr)),
      usable_records=[r['record']for r in rr if r['eligible']],range_pairs=sorted({str(r['ranges'])for r in rr})))
assert all(file_hash(r['path'])==r['sha256']and file_hash(Path(r['path']).parent/'metadata.json')==r['metadata_sha256']for r in rows)
d=dict(total=len(rows),usable=sum(r['eligible']for r in rows),summary=summary,records=rows)
(DEST/'第36至45次检查.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
with(DEST/'第36至45次检查.csv').open('w',encoding='utf-8-sig',newline='')as f:
    w=csv.writer(f);w.writerow(['布料','记录','可用','报错或原因','去程X范围mm','回程X范围mm','配置起点','采集时间','CSV路径'])
    for r in rows:w.writerow([r['label'],r['record'],'是'if r['eligible']else'否',r['reason'],str(r['ranges']['step_003']),str(r['ranges']['step_004']),str(r['saved_start']),r['created_at'],r['path']])
print('TOTAL',d['total'],'USABLE',d['usable'])
for s in summary:print('CLASS',s)
