"""Copy only confirmed rule-failing records for review; leave originals intact."""
import json,shutil,sys
from pathlib import Path
DEST=Path(__file__).parent;ROOT=DEST.parents[2];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import file_hash
d=json.loads((DEST/'质量与去回一致性结果.json').read_text());items=[]
for row in d['records']:
    if not row['action'].startswith('不合格'):continue
    source=Path(row['path']);metadata=source.parent/'metadata.json'
    assert file_hash(source)==row['sha256']and file_hash(metadata)==row['metadata_sha256'],str(source)
    target=DEST/'不合格数据副本'/row['label']/row['record'];target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target/source.name);shutil.copy2(metadata,target/'metadata.json')
    assert file_hash(target/source.name)==row['sha256']and file_hash(target/'metadata.json')==row['metadata_sha256']
    items.append(dict(record=row['record'],label=row['label'],original_csv=str(source),copy_csv=str(target/source.name),sha256=row['sha256'],metadata_sha256=row['metadata_sha256'],reason=row['reasons']))
assert len(items)==38
(DEST/'不合格数据副本/副本来源清单.json').write_text(json.dumps(dict(count=len(items),original_files_unchanged=True,items=items),ensure_ascii=False,indent=2))
print('COPIED AND HASH-VERIFIED',len(items),'CSV + metadata pairs; originals preserved.')
