import pytest
import torch
from cloth_trainer.config import TrainConfig

torch.set_num_threads(2)

@pytest.mark.parametrize('architecture', ['gru','tcn','cera'])
@pytest.mark.parametrize('readout', ['mean','last'])
def test_valid_segment_ignores_padding(architecture,readout):
    from cloth_trainer import classifiers
    torch.manual_seed(4)
    net=classifiers.SegmentClassifier(2,3,architecture,readout).eval()
    short=torch.randn(1,37,2)
    padded=torch.cat((short,torch.randn(1,19,2)*1000),1)
    with torch.no_grad():
        expected=net(short,torch.tensor([37]))
        actual=net(padded,torch.tensor([37]))
    assert expected.shape==(1,3)
    assert torch.allclose(actual,expected,atol=1e-5)
    assert not any('risk' in name for name,_ in net.named_parameters())
    assert hasattr(net,'gru')==(architecture in ('gru','cera'))
    assert hasattr(net,'tcn')==(architecture in ('tcn','cera'))
    assert hasattr(net,'retain')==(architecture=='cera')

@pytest.mark.parametrize('architecture', ['gru','tcn','cera'])
def test_features_are_causal_and_final_loss_updates_active_branches(architecture):
    from cloth_trainer import classifiers
    torch.manual_seed(5);net=classifiers.SegmentClassifier(2,3,architecture).eval()
    x=torch.randn(2,57,2);future=x.clone();future[:,29:]*=100
    a=net.features(x);b=net.features(future)
    assert torch.allclose(a[:,:29],b[:,:29],atol=1e-5)
    logits=net(x,torch.tensor([57,41]))
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]));loss.backward()
    assert net.class_head.weight.grad.abs().sum()>0
    if hasattr(net,'gru'):assert net.gru.weight_ih_l0.grad.abs().sum()>0
    if hasattr(net,'tcn'):assert net.tcn[0].conv.weight.grad.abs().sum()>0

def test_training_defaults_and_presets_keep_split_fixed():
    cfg=TrainConfig()
    assert cfg.architectures==('gru','tcn','cera') and cfg.signals==('by',)
    assert (cfg.epochs,cfg.patience,cfg.batch_size)==(200,30,16)
    quick=cfg.with_preset('quick');repeat=cfg.with_preset('repeat')
    assert (quick.epochs,quick.patience,quick.repetitions)==(30,8,1)
    assert repeat.repetitions==3 and repeat.split_seed==quick.split_seed==cfg.split_seed
    assert TrainConfig.from_dict(repeat.to_dict())==repeat
