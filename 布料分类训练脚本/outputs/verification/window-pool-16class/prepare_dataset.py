from pathlib import Path
from collections import Counter
from dataclasses import replace
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.dataset import derive_dataset,load_dataset
from cloth_trainer.data import file_hash
from cloth_trainer.modern_engine import frozen
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.tcn_models import default_spec
from cloth_trainer.storage import write_json,read_json
SOURCE=ROOT/'outputs/models/CLOTH4_20261119_171027_2dfe99'
DEST=ROOT/'outputs/datasets/前20次_16类_不含印花丝绒_v1'
meta,cfg,old=frozen(SOURCE)
protected=[SOURCE/'metadata.json',SOURCE/'execution.json',SOURCE/'dataset/dataset.json',SOURCE/'dataset/dataset.sha256']+list(SOURCE.glob('*/model.pt'))
old_hashes={str(p):file_hash(p) for p in protected}
removed=[r for r in old if r.label=='印花丝绒']
raw_hashes={str(p):file_hash(p) for r in removed for p in (Path(r.path),Path(r.path).parent/'metadata.json')}
if not DEST.exists():derive_dataset(SOURCE/'dataset',DEST,['印花丝绒'],split_plan=meta['split_plan'])
records,manifest=load_dataset(DEST)
assert len(records)==310 and len({r.label for r in records})==16
assert all(r.label!='印花丝绒' for r in records)
assert len(list((DEST/'arrays').glob('*.npz')))==310
kept=[i for i,r in enumerate(old) if r.label!='印花丝绒'];mapping={a:b for b,a in enumerate(kept)}
assert manifest['split_plan']['test']==[mapping[i] for i in meta['split_plan']['test'] if i in mapping]
for oldfold,newfold in zip(meta['split_plan']['cv'],manifest['split_plan']['cv']):
    for key in ['train','val']:assert newfold[key]==[mapping[i] for i in oldfold[key] if i in mapping]
assert all(file_hash(p)==h for p,h in old_hashes.items())
assert all(file_hash(p)==h for p,h in raw_hashes.items())
config=replace(cfg,tcn_mode='window_pool',tcn_spec=default_spec(128),tcn_window=900,name='前20次 · 16类（不含印花丝绒）· 窗口与汇总对照')
config.validate()
project=dict(format='cloth-project-v3',config=config.to_dict(),store=str(ROOT/'outputs/models'),snapshot=str(DEST),overrides={},
             sources=[dict(path=r.path,selected=True,sha256=r.sha256,metadata_sha256=r.metadata_sha256) for r in records])
write_json(ROOT/'outputs/当前训练项目.json',project)
write_json(DEST.parent/'前20次_16类_训练项目.json',project)
verification=dict(dataset=str(DEST),dataset_id=manifest['dataset_id'],records=len(records),trials=sum(len(r.trials)for r in records),classes=dict(Counter(r.label for r in records)),
                  excluded_records=len(removed),development=len(manifest['split_plan']['dev']),test=len(manifest['split_plan']['test']),
                  folds=[{k:len(v) for k,v in f.items()}for f in manifest['split_plan']['cv']],old_models_and_snapshot_unchanged=True,removed_raw_sources_unchanged=True,
                  current_project=str(ROOT/'outputs/当前训练项目.json'),training_started=False,epochs=config.epochs,mode=config.tcn_mode)
write_json(ROOT/'outputs/verification/window-pool-16class/数据集验证.json',verification)
print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)
