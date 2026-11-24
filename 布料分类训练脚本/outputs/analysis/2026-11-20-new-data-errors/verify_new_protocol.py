import sys,json,collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.config import DataConfig
from cloth_trainer.data import load_record
D=Path(__file__).parent
CFG=DataConfig(out_low=51.,out_high=140.,back_low=30.,back_high=120.,contact_rule='fixed-x-v1')
def check(row):
 try:
  r=load_record(row['path'],CFG,signals=('by',))
  row['screenshot_fixed_rule']=dict(status=r.quality,reason='；'.join(r.quality_notes),bounds=[t.bounds for t in r.trials],points=[len(t.values) for t in r.trials])
 except Exception as e:row['screenshot_fixed_rule']=dict(status='data_error',reason=str(e))
 return row
if __name__=='__main__':
 rows=json.loads((D/'scan.json').read_text())['records'];rr=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for row in pool.map(check,rows):
   rr.append(row)
   if len(rr)%100==0:print('检查',len(rr),'/',len(rows),flush=True)
 summary=dict(statuses=dict(collections.Counter(r['screenshot_fixed_rule']['status'] for r in rr)),errors=dict(collections.Counter(r['screenshot_fixed_rule']['reason'] for r in rr if r['screenshot_fixed_rule']['status']=='data_error')),
 old_error_now_loaded=sum(r['fixed-x-v1']['status']=='data_error' and r['screenshot_fixed_rule']['status']!='data_error' for r in rr),old_loaded_now_error=sum(r['fixed-x-v1']['status']!='data_error' and r['screenshot_fixed_rule']['status']=='data_error' for r in rr))
 (D/'截图新协议复查.json').write_text(json.dumps(dict(data=CFG.__dict__,summary=summary,records=rr),ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
