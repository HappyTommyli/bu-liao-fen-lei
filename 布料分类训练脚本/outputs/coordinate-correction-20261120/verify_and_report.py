import sys,json,csv,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from cloth_trainer.config import DataConfig
from cloth_trainer.data import load_record
D=Path(__file__).parent
CFG=DataConfig(out_low=51,out_high=140,back_low=30,back_high=120,contact_rule='fixed-x-v1')
def check(r):
 if r['status']=='已平移':
  original=json.loads(Path(r['backup_metadata']).read_bytes());current=json.loads(Path(r['metadata_path']).read_bytes());current.pop('coordinate_correction');assert original==current
  assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['after_sha256']
  assert r['verification']['non_x_values_unchanged']
  try:
   rec=load_record(r['path'],CFG,signals=('by',));r['post_status']=rec.quality;r['post_reason']='；'.join(rec.quality_notes)
   assert [tuple(t.bounds) for t in rec.trials]==[(51,140),(30,120)]
  except (ValueError,OSError) as e:r['post_status']='data_error';r['post_reason']=str(e)
 return r
if __name__=='__main__':
 doc=json.loads((D/'applied.json').read_text());rows=doc['records']
 with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,rows))
 modified=[r for r in rows if r['status']=='已平移'];unique=len({r['before_sha256'] for r in modified})
 for oldhash in {r['before_sha256'] for r in modified}:
  group=[r for r in modified if r['before_sha256']==oldhash]
  assert len({r['after_sha256'] for r in group})==1,'原始内容相同的副本修改后不一致'
 doc['records']=rows;doc.update(unique_original_csv_count=unique,post_rules=CFG.__dict__,post_statuses=dict(Counter(r['post_status'] for r in modified)),new_start_counts=dict(Counter(str(r['new_start_mm']) for r in modified)))
 (D/'verified.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
 headers=['目录','布料','文件','原起点_mm','新起点_mm','原终点_mm','新终点_mm','平移量_mm','原基线位置_mm','新基线位置_mm','新边界复查状态','新边界复查原因','修改后CSV路径','原始CSV备份路径','修改前CSV_SHA256','修改后CSV_SHA256','其他字段核对','平移脉冲数','改动单元格数','修改前metadata_SHA256','修改后metadata_SHA256']
 vals=[[r['source'],r['label'],r['record'],r['old_start_mm'],r['new_start_mm'],r['old_end_mm'],r['new_end_mm'],r['shift_mm'],r['baseline_mm'],r['actual_new_baseline_mm'],r['post_status'],r['post_reason'],r['path'],r['backup_csv'],r['before_sha256'],r['after_sha256'],'逐行一致',r['shift_pulses'],r['verified_cells_shifted'],r['metadata_before_sha256'],r['metadata_after_sha256']] for r in modified]
 for name,h,v in [('坐标平移修改记录.csv',headers,vals)]:
  with (D/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(h);w.writerows(v)
 allheaders=['目录','布料','文件','处理','处理原因','原基线_mm','原起点_mm','原终点_mm','新起点_mm','新终点_mm','CSV路径','备份路径']
 allvals=[[r['source'],r['label'],r['record'],r['status'],r['reason'],r.get('baseline_mm'),r.get('old_start_mm'),r.get('old_end_mm'),r.get('new_start_mm',r.get('old_start_mm')),r.get('new_end_mm',r.get('old_end_mm')),r['path'],r.get('backup_csv','')] for r in rows]
 with (D/'全部CSV检查记录.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(allheaders);w.writerows(allvals)
 inp=dict(headers=headers,modified=vals,allheaders=allheaders,allrecords=allvals,summary=doc['summary'],unique=unique,post_statuses=doc['post_statuses'],counts=dict(scanned=len(rows),modified=len(modified),unchanged=len(rows)-len(modified)),backup_root=doc['backup_root'])
 (D/'workbook-input.json').write_text(json.dumps(inp,ensure_ascii=False))
 print(json.dumps({k:doc[k] for k in ['summary','unique_original_csv_count','post_statuses','new_start_counts']},ensure_ascii=False))
