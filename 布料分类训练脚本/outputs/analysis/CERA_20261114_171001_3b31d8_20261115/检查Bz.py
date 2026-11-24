"""Read-only Bz audit on the exact frozen training windows. No unwrap/repair."""
import sys,csv,json,collections
from pathlib import Path
import numpy as np
ROOT=Path('/home/serein/classify-cloth/布料分类训练脚本');sys.path.insert(0,str(ROOT))
from cloth_trainer.data import load_record
from cloth_trainer.config import TrainConfig
m=json.loads((ROOT/'outputs/models/CERA_20261114_171001_3b31d8/metadata.json').read_text());cfg=TrainConfig.from_dict(m['config']);results=[];jumps=[]
for s in m['sources']:
 r=load_record(s['path'],cfg.data,s['label']);stages={'step_003':[],'step_004':[]}
 with open(s['path'],encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   if row['stage'] in stages and row['sample_valid']=='1':
    v=[float(row[k]) for k in ['sensor_t_s','motor_x_pulses','Bx_uT','By_uT','Bz_uT','dBz_uT']]
    if np.isfinite(v).all():stages[row['stage']].append(v)
 for t in r.trials:
  st='step_003' if t.direction=='out' else 'step_004';a=np.asarray(stages[st]);a=a[(a[:,1]/r.ppmm>=t.bounds[0])&(a[:,1]/r.ppmm<=t.bounds[1])];pos=np.searchsorted(a[:,0],t.times,side='right')-1;assert pos.min()>=0;v=a[pos];z=v[:,4];dz=v[:,5];jump=np.abs(np.diff(z));jj=np.flatnonzero(jump>10000)
  for i in jj:jumps.append({'path':s['path'],'direction':t.direction,'t':float(t.times[i+1]-t.times[0]),'before':float(z[i]),'after':float(z[i+1]),'jump':float(jump[i])})
  results.append({'path':s['path'],'label':s['label'],'direction':t.direction,'bz_min':float(z.min()),'bz_max':float(z.max()),'dbz_min':float(dz.min()),'dbz_max':float(dz.max()),'dbz_std':float(dz.std()),'large_jumps':len(jj),'bz_negative_fraction':float((z<0).mean())})
summary={'csv':len(m['sources']),'trials':len(results),'trials_with_jump':sum(x['large_jumps']>0 for x in results),'csv_with_jump':len({x['path'] for x in results if x['large_jumps']}),'total_jumps':len(jumps),'median_jump':float(np.median([x['jump'] for x in jumps])) if jumps else None,'by_label':{label:{'trials_with_jump':sum(x['label']==label and x['large_jumps']>0 for x in results),'jumps':sum(x['large_jumps'] for x in results if x['label']==label)} for label in m['labels']}}
(Path(__file__).parent/'Bz有效段质量核查.json').write_text(json.dumps({'summary':summary,'trials':results,'jumps':jumps},ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2));print('Examples',json.dumps(jumps[:3],ensure_ascii=False))
for direction in ['out','back']:
 rr=[x for x in results if x['direction']==direction];print(direction,'Bz min/max',min(x['bz_min'] for x in rr),max(x['bz_max'] for x in rr),'dBz min/max',min(x['dbz_min'] for x in rr),max(x['dbz_max'] for x in rr))
