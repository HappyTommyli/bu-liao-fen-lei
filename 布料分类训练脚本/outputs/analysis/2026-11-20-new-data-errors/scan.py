import sys,json,collections,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.config import DataConfig
from cloth_trainer.data import load_record,file_hash
DEST=Path(__file__).parent

def check(path):
 p=Path(path);meta=json.loads((p.parent/'metadata.json').read_text());row=dict(path=str(p),record=p.stem,label=meta.get('label',''),created_at=meta.get('created_at'),sequence=meta.get('sequence'),sha256=file_hash(p))
 for rule in ('fixed-x-v1','relative-035-v1'):
  try:
   r=load_record(p,DataConfig(contact_rule=rule),signals=('by',))
   row[rule]=dict(status=r.quality,reason='；'.join(r.quality_notes),points=[len(t.values) for t in r.trials],bounds=[list(t.bounds) for t in r.trials])
  except Exception as e:row[rule]=dict(status='data_error',reason=str(e))
 return row

if __name__=='__main__':
 paths=sorted((ROOT.parent/'xz-control/data/布料采集数据新').rglob('*.csv'))
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for row in pool.map(check,paths):
   rows.append(row)
   if len(rows)%50==0:print('已检查',len(rows),'/',len(paths),flush=True)
 result=dict(source=str(paths[0].parents[1]),files=len(rows),rules={rule:dict(statuses=dict(collections.Counter(r[rule]['status'] for r in rows)),errors=dict(collections.Counter(r[rule]['reason'] for r in rows if r[rule]['status']=='data_error'))) for rule in ('fixed-x-v1','relative-035-v1')},records=rows)
 (DEST/'scan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (DEST/'逐样本错误与规则对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['文件','布料','固定X状态','固定X原因','相对起点状态','相对起点原因','原始路径'])
  for r in rows:w.writerow([r['record'],r['label'],r['fixed-x-v1']['status'],r['fixed-x-v1']['reason'],r['relative-035-v1']['status'],r['relative-035-v1']['reason'],r['path']])
 print(json.dumps(result['rules'],ensure_ascii=False),flush=True)
