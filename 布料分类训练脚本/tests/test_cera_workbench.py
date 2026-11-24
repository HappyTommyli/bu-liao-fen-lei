from dataclasses import replace
import pytest
import torch
from test_data import fixture_csv
from cloth_trainer.tcn_ui import focused_config
from cloth_trainer.tcn_models import default_spec


def cera_config(**kw):
    from cloth_trainer.cera_models import default_cera_spec
    kw.setdefault('tcn_mode','single');kw.setdefault('tcn_stride',0)
    return replace(focused_config(),architectures=('cera',),tcn_spec=default_cera_spec(16,1,gru_hidden=24),tcn_window=128,device='cpu',workers=0,epochs=1,batch_size=8,**kw).validate()


def test_cera_fusion_is_causal_padding_safe_and_trainable():
    from cloth_trainer.cera_models import CERAClassifier,default_cera_spec
    model=CERAClassifier(1,3,default_cera_spec(16,2,gru_hidden=24,gru_layers=2)).eval()
    x=torch.randn(2,128,1)
    z=model.features(x);changed=x.clone();changed[:,64:]+=100
    assert torch.allclose(z[:,:64],model.features(changed)[:,:64],atol=1e-6)
    y=model(x,torch.tensor([128,64]));changed[0]=x[0]
    assert torch.allclose(y[1],model(changed,torch.tensor([128,64]))[1],atol=1e-6)
    y.sum().backward()
    for layer in (model.gru,model.blocks,model.gate,model.retain,model.add):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.parameters())
    model(torch.randn_like(x),torch.tensor([128,128]))
    assert torch.allclose(y.detach(),model(x,torch.tensor([128,64])),atol=1e-6)


def test_cera_candidate_cannot_be_loaded_as_tcn():
    from cloth_trainer.modern_training import make_model
    from cloth_trainer.tcn_study import candidates
    cfg=cera_config();candidate=candidates(cfg)[0]
    with pytest.raises(ValueError):make_model(cfg,1,2,'tcn','mean',candidate)
    with pytest.raises(ValueError):replace(cfg,architectures=('tcn',)).validate()


def test_cera_paired_lifecycle_final_online_and_replay(tmp_path):
    from cloth_trainer.engine import train_request,final_request,load_bundle,predict_request
    from cloth_trainer.modern_engine import load_online_bundle
    from cloth_trainer.storage import read_json
    from cloth_trainer.online_ui import OnlineWorker
    from cloth_trainer.online import OnlineConfig
    cfg=cera_config(tcn_mode='overlap_pair',tcn_stride=64)
    rr=[__import__('cloth_trainer.data',fromlist=['load_record']).load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(3)]
    task=train_request(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],config=cfg.to_dict(),output=str(tmp_path/'models')))
    meta=read_json(task/'metadata.json');assert len(meta['models'])==2
    key=next(iter(meta['models']));path=task/key/'model.pt';bundle=load_online_bundle(path)
    assert bundle['architecture']=='cera' and bundle['_model'].spec['gru_hidden']==24
    assert len(meta['contrasts']['B-A']['folds'])==1
    val=bundle['sources'][bundle['split_audit']['val'][0]]['path']
    worker=OnlineWorker(path,[val],OnlineConfig(window=128,stride=64,probability=0,margin=0,stable_seconds=0),source='validation-tuning')
    events=[];errors=[];worker.observed.connect(events.append);worker.failed.connect(errors.append);worker.run()
    assert not errors and any(e.get('type')=='prediction' for e in events)
    final_request(dict(task=str(task),configuration_id=meta['models'][key]['configuration_id']))
    assert load_bundle(task/'final/model.pt')['_model'].spec['gru_hidden']==24
    destination=tmp_path/'pred.json';predict_request(dict(models=[str(task/'final/model.pt')],paths=[val],destination=str(destination)))
    assert read_json(destination)['results'][0]['predictions']


def test_online_rejects_old_cera_without_window_candidate():
    from cloth_trainer.online import OnlinePredictor,OnlineConfig
    with pytest.raises(ValueError):OnlinePredictor(dict(architecture='cera',config={}),labels=['a','b'],predict_proba=lambda x:[.8,.2],config=OnlineConfig())


def test_legacy_whole_record_tcn_has_no_online_window_definition():
    from cloth_trainer.online import OnlinePredictor,OnlineConfig
    cfg=replace(focused_config(),tcn_spec=None)
    with pytest.raises(ValueError):OnlinePredictor(dict(architecture='tcn',config=cfg.to_dict()),labels=['a','b'],predict_proba=lambda x:[.8,.2],config=OnlineConfig())
