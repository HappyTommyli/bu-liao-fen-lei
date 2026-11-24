"""Refresh only new/changed files after the directory scan; preserve read-only audit."""
import runpy,json,hashlib,time,csv
from pathlib import Path
m=runpy.run_path(str(Path(__file__).with_name('audit.py')))
OUT=m['OUT'];result=json.loads((OUT/'audit.json').read_text());records={r['path']:r for r in result['records']};changed=[]
paths,ignored,errors=m['discover_report']([m['SOURCE']])
for p in paths:
 old=records.get(str(p));raw=p.read_bytes();mp=p.parent/'metadata.json';mb=mp.read_bytes() if mp.exists() else None
 sha=hashlib.sha256(raw).hexdigest();mh=hashlib.sha256(mb).hexdigest() if mb else None
 if old and old['csv_sha256']==sha and old['metadata_sha256']==mh:continue
 meta=json.loads(mb) if mb else {}
 with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 new,detail=m['adjust'](rows)
 r=dict(path=str(p),name=p.stem,label=meta.get('label',p.stem.rsplit('_',1)[0]),created_at=meta.get('created_at'),outcome=meta.get('outcome'),rows=len(rows),csv_sha256=sha,metadata_sha256=mh,quality_mask=detail,checks={})
 for rule,cfg in m['CONFIGS'].items():r['checks'][rule]=dict(original=m['check'](p,cfg,rows,fields),by_only=m['check'](p,cfg,new,fields))
 r['unchanged_after_check']=(p.read_bytes()==raw and (mp.read_bytes() if mp.exists() else None)==mb)
 records[str(p)]=r;changed.append(p.stem)
result.update(records=[records[str(p)] for p in paths],ignored=ignored,discovery_errors=errors,refreshed_at=time.strftime('%Y-%m-%d %H:%M:%S%z'),refreshed_files=changed)
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('refreshed',changed,'total',len(records))
