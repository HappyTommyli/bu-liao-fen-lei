import sys, json, csv, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from cloth_trainer.data import discover_report,import_dataset,file_hash
from cloth_trainer.config import DataConfig
SOURCE=ROOT.parent/'xz-control/data/布料采集数据'
DEST=Path(__file__).parent
paths,_,_=discover_report([SOURCE]);selected=[]
for p in paths:
    m=json.loads((p.parent/'metadata.json').read_text(encoding='utf-8-sig'))
    if 35<=int(m.get('sequence') or 0)<=45 and m.get('label') not in ('三角形间隔','矩形间隔'):
        selected.append(p)
before={str(p):(file_hash(p),file_hash(p.parent/'metadata.json')) for p in selected}
def progress(i,n):
    if i%25==0 or i==n: print(f'REVIEW {i}/{n}',flush=True)
records,report=import_dataset(selected,DataConfig(),signals=('by',),progress=progress)
rows=[]
for r in records:
    d=r.diagnostics['contact_rule']
    rows.append(dict(record=Path(r.path).stem,label=r.label,path=r.path,status=r.quality,eligible=r.audit['eligible'],reason='；'.join(r.audit['reasons']),contact=d,points=[len(t.values) for t in r.trials]))
for e in report['errors']:
    rows.append(dict(record=Path(e['path']).stem,path=e['path'],status=e.get('audit',{}).get('status','error'),eligible=False,reason=e['message']))
assert all((file_hash(p),file_hash(Path(p).parent/'metadata.json'))==hashes for p,hashes in before.items())
summary=dict(total=len(selected),loaded=len(records),eligible=sum(r['eligible'] for r in rows),statuses=dict(collections.Counter(r['status'] for r in rows)),sources_unchanged=True)
(DEST/'新接触窗审查.json').write_text(json.dumps(dict(summary=summary,records=rows),ensure_ascii=False,indent=2))
(DEST/'审查批次.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (DEST/'新接触窗审查.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['记录','可训练','状态','原因','起点Xmm','绝对接触窗Xmm','CSV路径'])
    for r in rows:
        d=r.get('contact',{})
        w.writerow([r['record'],r['eligible'],r['status'],r['reason'],d.get('origin_x_mm'),d.get('absolute_x_mm'),r['path']])
print(json.dumps(summary,ensure_ascii=False),flush=True)
for r in rows:
    if not r['eligible']: print(r['record'],r['reason'],flush=True)
