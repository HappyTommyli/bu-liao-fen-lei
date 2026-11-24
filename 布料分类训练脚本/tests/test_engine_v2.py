import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from cloth_trainer.config import TrainConfig
from cloth_trainer.data import load_record,prepare_trials,file_hash
from cloth_trainer.engine import train_request,predict_request,load_bundle,Cancelled
from cloth_trainer.storage import read_json,export_task,import_task
from test_data import fixture_csv

@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root=tmp_path_factory.mktemp('source_v2')
    return [load_record(fixture_csv(root,label,i,speed=100)) for label in ['布A','布B'] for i in range(10)]

def request(source,out,signals=('by',),**kw):
    cfg=TrainConfig(epochs=1,signals=signals,batch_size=8,**kw)
    return dict(paths=[r.path for r in source],sources=[r.manifest() for r in source],overrides={r.path:r.label for r in source},config=cfg.to_dict(),output=str(out))

def test_fit_does_not_test_or_calibrate_and_scaler_uses_train_only(source,tmp_path):
    events=[];task=train_request(request(source,tmp_path/'out'),events.append);m=read_json(task/'metadata.json')
    assert m['format']=='cloth-classification-v3' and m['status']=='completed' and len(m['models'])==15
    assert len(m['partitions'])==5 and all(set(p)=={'train','val','test'} for p in m['partitions'])
    for key,r in m['models'].items():
        train=[load_record(m['sources'][i]['path']) for i in m['partitions'][r['fold']]['train']]
        expected=np.concatenate([t.values for t in prepare_trials(train,'by')]).mean(0)
        assert r['test_metrics'] is None and r['test_predictions']==[]
        assert 'calibration' not in r and 'prefix_metrics' not in r
        assert 'risk_loss' not in r['history'][0]
        assert set(r['val_direction_metrics'])=={'out','back'}
        assert r['validation_inference_seconds']>=0
        bundle=load_bundle(task/key/'model.pt')
        assert np.allclose(bundle['scaler']['mean'],expected,rtol=1e-5)
        assert bundle['model_id']==r['model_id']
        assert not (task/key/'calibration.json').exists()
    assert any(e.get('type')=='epoch' for e in events)

def test_manual_test_portable_bundle_and_prediction(source,tmp_path):
    from cloth_trainer.engine import test_request
    task=train_request(request(source,tmp_path/'out',('by','bxby','xy'),architectures=('tcn',)))
    before=read_json(task/'metadata.json');hashes={k:file_hash(task/k/'model.pt') for k in before['models']}
    test_request(dict(task=str(task)))
    m=read_json(task/'metadata.json')
    assert all(r['test_metrics']['total']==8 for r in m['models'].values())
    assert all(hashes[k]==file_hash(task/k/'model.pt') for k in m['models'])
    key=next(iter(m['models']));dest=tmp_path/'prediction.json'
    predict_request(dict(models=[str(task/key/'model.pt')],paths=[source[0].path],destination=str(dest)))
    rows=read_json(dest)['results'][0]['predictions']
    assert len(rows)==2 and all(r['training_overlap'] for r in rows)
    assert all('predicted' in r and 'risk' not in r for r in rows)
    archive=tmp_path/'package.zip';export_task(task,archive);imported=import_task(archive,tmp_path/'imported')
    assert load_bundle(imported/key/'model.pt')['model_id']==m['models'][key]['model_id']
    original=Path(source[0].path);newdir=tmp_path/'new';newdir.mkdir();newcsv=newdir/'new.csv';shutil.copy(original,newcsv)
    md=read_json(original.parent/'metadata.json');md['context']['config']['x_speed']=101
    (newdir/'metadata.json').write_text(json.dumps(md))
    predict_request(dict(models=[str(task/key/'model.pt')],paths=[str(newcsv)],destination=str(dest)))
    r=read_json(dest)['results'][0];assert not r['predictions'] and any('协议' in e['message'] for e in r['errors'])

def test_repetitions_keep_split_and_configuration_identity(source,tmp_path):
    task=train_request(request(source,tmp_path/'out',architectures=('gru',),repetitions=3))
    rows=list(read_json(task/'metadata.json')['models'].values())
    assert len(rows)==15 and {r['seed'] for r in rows}=={42,43,44}
    assert len({r['configuration_id'] for r in rows})==1 and len({r['model_id'] for r in rows})==15

def test_cancelled_and_tampered_models_are_not_loadable(source,tmp_path):
    with pytest.raises(Cancelled):train_request(request(source,tmp_path/'cancel'),cancelled=lambda:True)
    assert read_json(next((tmp_path/'cancel').glob('*/metadata.json')))['status']=='cancelled'
    task=train_request(request(source,tmp_path/'out',architectures=('gru',)))
    m=read_json(task/'metadata.json');key=next(iter(m['models']));p=task/key/'model.pt';p.write_bytes(p.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='权重|完整'):load_bundle(p)

def test_manual_test_uses_frozen_holdout_and_rejects_changed_snapshot(source,tmp_path):
    from cloth_trainer.engine import test_request
    task=train_request(request(source,tmp_path/'out',architectures=('gru',)))
    m=read_json(task/'metadata.json');p=Path(m['sources'][m['partitions'][0]['test'][0]]['path']);raw=p.read_bytes()
    try:
        p.write_bytes(raw+b'\n')
        test_request(dict(task=str(task)))  # frozen processed records remain unchanged
        a=next((task/'dataset/arrays').glob('*.npz'));a.write_bytes(a.read_bytes()+b'changed')
        with pytest.raises(ValueError,match='校验'):test_request(dict(task=str(task)))
    finally:p.write_bytes(raw)


def test_quick_trial_is_not_formal_and_test_candidates_are_locked(source,tmp_path):
    from cloth_trainer.engine import test_request
    quick=train_request(request(source,tmp_path/'quick',architectures=('tcn',),preset='quick'))
    qm=read_json(quick/'metadata.json');key=next(iter(qm['models']))
    with pytest.raises(ValueError,match='快速'):load_bundle(quick/key/'model.pt')
    with pytest.raises(ValueError,match='快速'):test_request({'task':str(quick)})
    task=train_request(request(source,tmp_path/'out',architectures=('gru','tcn')))
    keys=list(read_json(task/'metadata.json')['models'])
    test_request({'task':str(task),'keys':keys[:1]})
    test_request({'task':str(task),'keys':keys[:1]})  # same frozen selection is idempotent
    with pytest.raises(ValueError,match='锁定'):test_request({'task':str(task),'keys':keys})
