import sys,json,csv,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from PyQt5 import QtCore
from cloth_trainer.storage import list_tasks,read_json
from cloth_trainer.online_ui import OnlineWorker
from cloth_trainer.online import OnlineConfig,csv_packets,summarize_records

out=Path(__file__).resolve().parent
torch.set_num_threads(2)
meta=next(m for m in list_tasks('outputs/models') if m.get('status')=='completed' and m.get('config',{}).get('architectures')==['cera'])
task=Path(meta['directory']);partition=meta['split_plan']['cv'][0]
paths=[meta['sources'][i]['path'] for i in partition['val']]
assert set(partition['val']).isdisjoint(meta['split_plan']['test'])
cfg=OnlineConfig(window=900,stride=900,probability=.95,margin=.20,stable_seconds=2.7)
print(json.dumps(dict(task=str(task),validation_files=len(paths),config=asdict(cfg)),ensure_ascii=False),flush=True)
result=dict(task=str(task),name=meta['name'],config=asdict(cfg),source='frozen validation; no holdout',models={})
bounds={}
for path in paths:
 frames=[p for p in csv_packets(path) if p.get('type')=='sample' and p.get('stage')=='step_003']
 active=[p for p in frames if 51<=p['x_mm']<=140]
 bounds[str(path)]=dict(start=active[0]['t_s'],end=active[-1]['t_s'],scan_end=frames[-1]['t_s'],x_start=active[0]['x_mm'],x_end=active[-1]['x_mm']) if active else {}
for key,row in meta['models'].items():
 started=time.perf_counter();worker=OnlineWorker(task/key/'model.pt',paths,cfg,source='validation-tuning');records=[];events=[];errors=[]
 worker.record.connect(records.append);worker.observed.connect(events.append);worker.failed.connect(errors.append)
 worker.run()
 if errors:raise RuntimeError(errors)
 for record in records:
  b=bounds[record['path']];first=record['first_confirmation'];record['crop_duration_s']=b['end']-b['start'];record['scan_end_t_s']=b['scan_end']
  record['first_from_crop_s']=None if first is None else first['t_s']-b['start']
  record['lead_before_crop_end_s']=None if first is None else b['end']-first['t_s']
  record['lead_before_scan_end_s']=None if first is None else b['scan_end']-first['t_s']
  record['first_correct']=None if first is None else first['label']==record['true_label']
  record['early_confirmed']=bool(first and first['t_s']<b['end'])
 confirmed=[r for r in records if r['first_confirmation']]
 summary=summarize_records(records)
 summary.update(validation_files=len(paths),early_confirmed=sum(r['early_confirmed'] for r in records),early_correct=sum(r['early_confirmed'] and r['first_correct'] for r in records),median_first_from_crop_s=float(np.median([r['first_from_crop_s'] for r in confirmed])) if confirmed else None,median_lead_crop_s=float(np.median([r['lead_before_crop_end_s'] for r in confirmed])) if confirmed else None,median_first_x_mm=float(np.median([r['first_confirmation']['x_mm'] for r in confirmed])) if confirmed else None,first_wrong=[dict(path=r['path'],true=r['true_label'],first=r['first_confirmation']) for r in confirmed if not r['first_correct']],median_crop_duration_s=float(np.median([r['crop_duration_s'] for r in records])),seconds=time.perf_counter()-started)
 result['models'][key]=dict(summary=summary,records=records,events=events)
 print(json.dumps(dict(key=key,summary=summary),ensure_ascii=False),flush=True)
 (out/'replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (out/(key+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  fields=['path','true_label','prediction','status','first_from_crop_s','lead_before_crop_end_s','lead_before_scan_end_s','first_correct','early_confirmed','crop_duration_s','flip_count','quality_resets'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
