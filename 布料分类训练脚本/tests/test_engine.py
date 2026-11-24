import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from cloth_trainer.config import TrainConfig,DataConfig
from cloth_trainer.data import load_record
from cloth_trainer.engine import train_request,predict_request,load_bundle,Cancelled
from cloth_trainer.storage import read_json,export_task,import_task
from test_data import fixture_csv

@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp('cera_source')
    return [load_record(fixture_csv(root,label,i,speed=100)) for label in ['布A','布B'] for i in range(10)]

def request(source,out,signals=('by',)):
    cfg=TrainConfig(epochs=1,signals=signals,batch_size=8)
    return dict(paths=[r.path for r in source],sources=[r.manifest() for r in source],
                overrides={r.path:r.label for r in source},config=cfg.to_dict(),output=str(out))

def test_cancelled_task_is_not_completed(source,tmp_path):
    with pytest.raises(Cancelled):train_request(request(source,tmp_path/'out'),cancelled=lambda:True)
    m=list((tmp_path/'out').glob('*/metadata.json'))
    assert read_json(m[0])['status']=='cancelled'

def test_changed_source_rejected_before_training(source,tmp_path):
    req=request(source,tmp_path/'out');req['sources'][0]['sha256']='stale'
    with pytest.raises(ValueError,match='改变'):train_request(req)

def test_metadata_change_invalidates_training_snapshot(source,tmp_path):
    req=request(source,tmp_path/'out');p=Path(source[0].path).parent/'metadata.json';original=p.read_bytes()
    changed=json.loads(original);changed['context']['capture_session']='different-capture'
    try:
        p.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='元数据.*改变'):train_request(req)
    finally:p.write_bytes(original)
