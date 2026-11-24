from dataclasses import replace
import pytest
import torch


def test_config_is_opt_in_and_preserves_legacy_serialization():
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.tcn_models import default_spec
    old=ModernConfig().to_dict()
    assert not {'tcn_spec','tcn_mode','tcn_window','grad_clip_norm','scheduler_factor'} & old.keys()
    assert ModernConfig.from_dict(old).to_dict()==old
    cfg=ModernConfig(architectures=('tcn',),tcn_spec=default_spec())
    assert ModernConfig.from_dict(cfg.to_dict()).to_dict()==cfg.to_dict()
    with pytest.raises(ValueError):replace(cfg,architectures=('gru',)).validate()
    with pytest.raises(ValueError):replace(cfg,signals=('bx',)).validate()
    with pytest.raises(ValueError):replace(cfg,tcn_mode='four_way',tcn_spec=default_spec(128)).validate()


@pytest.mark.parametrize('pooling',['mean','mean_std','max'])
def test_custom_model_is_causal_and_padding_invariant(pooling):
    from cloth_trainer.tcn_models import TCNClassifier,default_spec
    torch.manual_seed(3)
    spec=default_spec();spec.update(channels=[32,64,96],dilations=[1,3,5],kernel_size=3,pooling=pooling)
    model=TCNClassifier(1,3,spec).eval()
    x=torch.randn(2,40,1);lengths=torch.tensor([40,1])
    p=model(x,lengths)
    xx=torch.cat([x,torch.randn(2,15,1)*100],1)
    assert torch.allclose(p,model(xx,lengths),atol=1e-6)
    altered=x.clone();altered[:,20:]+=100
    assert torch.allclose(model.features(x)[:,:20],model.features(altered)[:,:20],atol=1e-6)
    assert torch.isfinite(p).all()
    assert model.receptive_field==19
    p.sum().backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


def test_model_default_parameter_count_and_validation():
    from cloth_trainer.tcn_models import TCNClassifier,default_spec,normalize_spec
    for width,total in [(64,104975),(128,414735)]:
        m=TCNClassifier(1,15,default_spec(width))
        assert sum(p.numel() for p in m.parameters())==total
        assert m.receptive_field==253
    for bad in [dict(kernel_size=4),dict(channels=[64,64]),dict(dropout=float('nan')),dict(dilations=[0]*6),dict(revision='unknown')]:
        with pytest.raises(ValueError):normalize_spec(dict(default_spec(),**bad))


@pytest.mark.parametrize('length,expected',[(5316,[900]*5+[816]),(5372,[900]*5+[872]),(40,[40]),(900,[900]),(1800,[900,900]),(1801,[900,900,1])])
def test_window_partition_preserves_every_point(length,expected):
    from cloth_trainer.tcn_study import window_slices,split_batch
    slices=window_slices(length,900)
    assert [b-a for a,b in slices]==expected
    assert [v for a,b in slices for v in range(a,b)]==list(range(length))
    x=torch.arange(length,dtype=torch.float32)[None,:,None]
    windows,lens,parents,weights,starts=split_batch(x,torch.tensor([length]),900)
    assert lens.tolist()==expected and parents.tolist()==[0]*len(expected)
    assert starts.tolist()==[a for a,b in slices]
    assert torch.allclose(weights,torch.tensor(expected)/length)
    assert torch.equal(torch.cat([w[:n] for w,n in zip(windows,lens)]),x[0])


def test_probability_aggregation_is_length_weighted_and_stable():
    from cloth_trainer.tcn_study import aggregate_log_probs
    logits=torch.tensor([[0.,0.],[0.,2.],[1000.,-1000.]])
    parents=torch.tensor([0,0,1]);weights=torch.tensor([.75,.25,1.])
    logp=aggregate_log_probs(logits,parents,weights,2)
    assert torch.allclose(logp[0].exp(),.75*logits[0].softmax(-1)+.25*logits[1].softmax(-1))
    assert torch.isfinite(logp).all() and logp[1,1]<-1900


def test_four_way_candidates_are_frozen_and_have_unique_ids():
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.tcn_models import default_spec
    from cloth_trainer.tcn_study import candidates,validate_candidate
    cfg=ModernConfig(architectures=('tcn',),tcn_spec=default_spec(),tcn_mode='four_way')
    cc=candidates(cfg)
    assert [c['id'] for c in cc]==['full_h64','full_h128','w900_h64','w900_h128']
    assert [c['window_size'] for c in cc]==[0,0,900,900]
    assert [c['spec']['channels'][0] for c in cc]==[64,128,64,128]
    for c in cc:assert validate_candidate(c)==c
    with pytest.raises(ValueError):validate_candidate(dict(cc[0],aggregation='majority'))
