from pathlib import Path
import shutil
import pytest
import numpy as np
from cloth_trainer.data import load_record
from cloth_trainer.config import TrainConfig
from cloth_trainer.dataset import save_dataset,load_dataset
from test_data import fixture_csv


def test_portable_snapshot_frozen_arrays_config_folds(tmp_path):
    records=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(5)]
    cfg=TrainConfig();p=save_dataset(tmp_path/'snapshot',records,cfg.data,42,screening={'errors':[]})
    shutil.rmtree(tmp_path/'raw');shutil.move(str(p),str(tmp_path/'moved'))
    restored,manifest=load_dataset(tmp_path/'moved')
    assert len(restored)==10 and manifest['folds'] and manifest['dataset_id']
    assert all(np.array_equal(a.trials[0].values,b.trials[0].values) for a,b in zip(records,restored))
    assert restored[0].quality==records[0].quality and restored[0].axes==('y',)
    with pytest.raises(ValueError,match='覆盖|存在'):save_dataset(tmp_path/'moved',records,cfg.data,42)
    array=next((tmp_path/'moved').glob('arrays/*.npz'));array.write_bytes(array.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='校验|改变'):load_dataset(tmp_path/'moved')


def test_joint_snapshot_preserves_independent_input_views(tmp_path):
    from cloth_trainer.data import prepare_trials
    from test_r3_data import rewrite
    records=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by','bxby')) for c in ('布A','布B') for i in range(5)]
    p=Path(records[0].path)
    rewrite(p,modify=lambda row:row.update(Bx_uT='nan',dBx_uT='nan') if row['stage']=='step_003' and .8<float(row['sensor_t_s'])<.804 else None)
    records[0]=load_record(p,signals=('by','bxby'));reference=prepare_trials([load_record(p,signals=('by',))],'by')[0].values.copy()
    snap=save_dataset(tmp_path/'snap',records,TrainConfig().data,signals=('by','bxby'));restored,_=load_dataset(snap)
    assert np.array_equal(prepare_trials([restored[0]],'by')[0].values,reference)


@pytest.mark.parametrize('kind',['excluded','undetermined','error','unselected'])
def test_snapshot_freeze_checks_full_batch_sources(tmp_path,kind):
    import json
    from cloth_trainer.data import import_dataset
    from test_r3_data import rewrite
    paths=[fixture_csv(tmp_path/'raw',c,i,speed=100) for c in ('布A','布B') for i in range(6 if c=='布A' else 5)]
    if kind=='excluded':rewrite(paths[0],modify=lambda row:row.update(dBy_uT=str(float(row['dBy_uT'])*2),By_uT=str(float(row['By_uT'])*2)) if row['stage']=='step_004' else None)
    elif kind=='undetermined':
        mp=paths[0].parent/'metadata.json';m=json.loads(mp.read_text());m['context']['baseline'].pop('sigma');mp.write_text(json.dumps(m))
    elif kind=='error':paths[0].write_text('not,a,raw,csv\n')
    records,report=import_dataset(paths,signals=('by',))
    chosen=[r for r in records if r.audit['eligible'] and r.path!=str(paths[0])]
    paths[0].write_text(paths[0].read_text(encoding='utf-8-sig')+'\n',encoding='utf-8-sig')
    with pytest.raises(ValueError,match='批次|来源|改变'):save_dataset(tmp_path/'snap',chosen,TrainConfig().data,screening=report)
    assert not (tmp_path/'snap').exists()
