from dataclasses import replace
import torch
import pytest
from cloth_trainer.cera_models import default_cera_spec
from cloth_trainer.tcn_ui import focused_config


def config():
    return replace(focused_config(),architectures=('cera',),tcn_spec=dict(default_cera_spec(16,1,gru_hidden=16),revision='configurable-cera-prefix-v1'),tcn_mode='prefix',tcn_window=128,tcn_stride=128,device='cpu',workers=0,epochs=1,batch_size=8,cv_folds=1).validate()


def test_prefix_samples_keep_origin_and_parent_weight():
    from cloth_trainer.tcn_study import candidates,split_candidate,validate_candidate
    cfg=config();c=candidates(cfg)[0];assert validate_candidate(c)==c
    x=torch.arange(2*512,dtype=torch.float32).reshape(2,512,1)
    windows,lens,parents,weights,starts=split_candidate(x,torch.tensor([512,300]),c)
    assert starts.tolist()==[0]*7 and lens.tolist()==[128,256,384,512,128,256,300]
    assert torch.equal(windows[2,:384],x[0,:384])
    assert all(weights[parents==p].sum()==pytest.approx(1) for p in (0,1))
    sampled=split_candidate(x,torch.tensor([512,300]),c,generator=torch.Generator().manual_seed(42))
    assert len(sampled[0])==4
    assert all(sampled[3][sampled[2]==p].sum()==pytest.approx(1) for p in (0,1))


def test_prefix_lifecycle_and_final_prediction_uses_last_prefix(tmp_path):
    from test_data import fixture_csv
    from cloth_trainer.data import load_record
    from cloth_trainer.engine import train_request,final_request,load_bundle
    from cloth_trainer.modern_engine import load_online_bundle
    from cloth_trainer.storage import read_json
    cfg=config();rr=[load_record(fixture_csv(tmp_path/'raw',c,i,speed=100),signals=('by',)) for c in ('布A','布B') for i in range(3)]
    task=train_request(dict(paths=[r.path for r in rr],sources=[r.manifest() for r in rr],config=cfg.to_dict(),output=str(tmp_path/'models')))
    meta=read_json(task/'metadata.json');assert len(meta['models'])==1
    key,row=next(iter(meta['models'].items()));b=load_online_bundle(task/key/'model.pt')
    assert b['_model'].spec['revision']=='configurable-cera-prefix-v1'
    for r in row['val_predictions']:
        assert all(w['start']==0 for w in r['windows'])
        assert r['scores']==pytest.approx(r['windows'][-1]['scores'])
    final_request(dict(task=str(task),configuration_id=row['configuration_id']))
    assert load_bundle(task/'final/model.pt')['candidate']['training_revision']=='parent-prefix-v1'


def test_fast_parameter_summary_matches_real_model():
    from cloth_trainer.model_summary import parameter_summary
    from cloth_trainer.modern_training import make_model
    from cloth_trainer.tcn_study import candidates
    cfg=config();c=candidates(cfg)[0];summary=parameter_summary(c['spec'],17)
    model=make_model(cfg,1,17,'cera','mean',c)
    assert summary['parameters']==sum(p.numel() for p in model.parameters())
