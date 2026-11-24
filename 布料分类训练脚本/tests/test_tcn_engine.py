from dataclasses import replace
import json
import pytest
from test_data import fixture_csv
from cloth_trainer.data import load_record,file_hash
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.tcn_models import default_spec
from cloth_trainer.storage import read_json,write_json,export_task,import_task


@pytest.mark.parametrize('mode,count',[('single',5),('four_way',20),('window_pool',20)])
def test_tcn_lifecycle_freezes_candidates_and_reuses_window_inference(tmp_path,mode,count):
    from cloth_trainer.engine import train_request,final_request,test_request,load_bundle,predict_request
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    spec=default_spec(128) if mode=='window_pool' else default_spec() if mode=='four_way' else default_spec(32,2)
    cfg=ModernConfig(device='cpu',workers=0,architectures=('tcn',),tcn_spec=spec,tcn_mode=mode,
                     tcn_window=900 if mode!='single' else 128,epochs=1,batch_size=4)
    task=train_request(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],config=cfg.to_dict(),output=str(tmp_path/'models')))
    m=read_json(task/'metadata.json')
    assert len(m['models'])==count and 'study' in m
    assert len(m['summaries'])==count//5
    assert all(r['candidate'] in m['study']['candidates'] for r in m['models'].values())
    assert all(r['val_window_metrics']['total']>=r['val_metrics']['total'] for r in m['models'].values())
    if mode=='four_way':assert len(m['contrasts'])==4
    if mode=='window_pool':assert len(m['contrasts'])==3
    # Explicitly pick the larger windowed model to catch loading the global hidden_size instead.
    selected='tcn_by_mean_wp_b_w900_meanstd' if mode=='window_pool' else 'tcn_by_mean_w900_h128' if mode=='four_way' else next(iter(m['summaries']))
    final_events=[]
    final_request(dict(task=str(task),configuration_id=selected),final_events.append);m=read_json(task/'metadata.json')
    if mode=='window_pool':
        phase=next(e for e in final_events if e['type']=='phase')
        assert phase['scope']=='final' and phase['configuration_id']==selected and phase['task']==str(task)
        assert all(e.get('scope')=='final' for e in final_events if e['type']=='epoch')
    b=load_bundle(task/'final/model.pt')
    assert b['candidate']==m['final_model']['candidate']
    assert b['_model'].spec['channels'][0]==(128 if mode!='single' else 32)
    if mode=='window_pool':assert b['_model'].spec['pooling']=='mean_std'
    test_request(dict(task=str(task)));m=read_json(task/'metadata.json')
    assert m['holdout']['metrics']['total']==len(m['split_plan']['test'])*2
    assert m['holdout']['roundtrip_metrics']['total']==len(m['split_plan']['test'])
    archive=tmp_path/'artifact.zip';export_task(task,archive);copy=import_task(archive,tmp_path/'imported')
    dest=tmp_path/'pred.json';predict_request(dict(models=[str(copy/'final/model.pt')],paths=[r.path for r in rr],destination=str(dest)))
    rows=read_json(dest)['results'][0]['predictions'];assert len(rows)==24
    assert all('windows' in r for r in rows)
    # Even when the checksum is updated, inconsistent candidate metadata must fail.
    m=read_json(task/'metadata.json');m['final_model']['candidate']['window_size']=256
    write_json(task/'metadata.json',m)
    with pytest.raises(ValueError,match='候选|关联'):load_bundle(task/'final/model.pt')


def test_changed_study_revision_is_rejected_before_training(tmp_path):
    from cloth_trainer.modern_engine import initialize,frozen
    rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(6)]
    cfg=ModernConfig(device='cpu',workers=0,architectures=('tcn',),tcn_spec=default_spec(32,2))
    task,m,_,_=initialize(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],config=cfg.to_dict(),output=str(tmp_path/'models')))
    e=read_json(task/'execution.json');e['study']['revision']='unknown';write_json(task/'execution.json',e)
    m['study']=e['study'];m['execution_sha256']=file_hash(task/'execution.json');write_json(task/'metadata.json',m)
    with pytest.raises(ValueError,match='版本|候选|研究'):frozen(task)
