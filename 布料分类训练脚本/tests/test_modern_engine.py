from dataclasses import replace
import pytest
from test_data import fixture_csv
from cloth_trainer.data import load_record
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.storage import read_json,write_json,export_task,import_task


def test_modern_cv_refit_holdout_and_portable_inference(tmp_path):
    from cloth_trainer.engine import train_request,test_request,final_request,load_bundle,predict_request
    records=[load_record(fixture_csv(tmp_path/'raw',label,i,speed=100)) for label in ('布A','布B') for i in range(6)]
    cfg=ModernConfig(device='cpu',workers=0,epochs=1,hidden_size=32,architectures=('gru',),batch_size=8)
    request=dict(paths=[r.path for r in records],sources=[r.manifest() for r in records],overrides={r.path:r.label for r in records},config=cfg.to_dict(),output=str(tmp_path/'models'))
    task=train_request(request);meta=read_json(task/'metadata.json')
    assert len(meta['models'])==5 and not meta.get('final_model')
    assert all(r['test_metrics'] is None for r in meta['models'].values())
    with pytest.raises(ValueError,match='最终'):test_request(dict(task=str(task)))
    final_request(dict(task=str(task)));meta=read_json(task/'metadata.json');final=meta['final_model']
    assert final['epochs']==1 and final['train_records']==len(meta['split_plan']['dev'])
    assert final['train_records']+len(meta['split_plan']['test'])==12
    b=load_bundle(task/'final/model.pt');assert b['phase']=='final'
    with pytest.raises(ValueError,match='最终'):load_bundle(task/next(iter(meta['models']))/'model.pt')
    test_request(dict(task=str(task)));first=read_json(task/'metadata.json')['holdout']
    test_request(dict(task=str(task)));assert read_json(task/'metadata.json')['holdout']==first
    assert first['metrics']['total']==len(meta['split_plan']['test'])*2
    assert first['roundtrip_metrics']['total']==len(meta['split_plan']['test'])
    with pytest.raises(ValueError,match='锁定'):final_request(dict(task=str(task),configuration_id='tcn_by_mean'))
    archive=tmp_path/'result.zip';export_task(task,archive);copy=import_task(archive,tmp_path/'imported');load_bundle(copy/'final/model.pt')
    dest=tmp_path/'pred.json';predict_request(dict(models=[str(copy/'final/model.pt')],paths=[r.path for r in records],destination=str(dest)))
    assert len(read_json(dest)['results'][0]['predictions'])==24
    execution=read_json(task/'execution.json');execution['plan']['test']=[];write_json(task/'execution.json',execution)
    with pytest.raises(ValueError,match='校验'):test_request(dict(task=str(task)))
