from dataclasses import replace
import numpy as np
import pytest
import torch
from cloth_trainer.data import Trial


def trials():
    t=np.arange(24)/333
    return [Trial(str(i),f'/toy/{i//2}.csv',f'h{i//2}',f'g{i//2}','A' if i<4 else 'B','out' if i%2==0 else 'back',
                  (np.sin(t*90)+i).astype(np.float32)[:,None],t,t,t,np.zeros((24,1)),(0,24)) for i in range(8)]


def test_cpu_is_explicit_and_cuda_missing_fails_honestly(monkeypatch):
    from cloth_trainer.devices import resolve_device
    assert resolve_device('cpu').type=='cpu'
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CUDA|驱动'):resolve_device('cuda')


@pytest.mark.parametrize('architecture',['gru','tcn','cera','conv_gru','conv_cera'])
def test_modern_classifier_padding_and_compression(architecture):
    from cloth_trainer.modern_models import ModernClassifier
    torch.manual_seed(4);m=ModernClassifier(1,2,architecture,'mean',32,.1).eval()
    x=torch.randn(2,25,1);lengths=torch.tensor([25,17]);p=m(x,lengths)
    assert p.shape==(2,2) and torch.isfinite(p).all()
    q=m(torch.cat([x,torch.randn(2,7,1)],1),lengths)
    assert torch.allclose(p,q,atol=1e-6)


def test_microbatch_weighting_matches_full_effective_batch(tmp_path):
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_training import fit
    torch.set_num_threads(1);tt=trials();cfg=ModernConfig(device='cpu',workers=0,epochs=2,batch_size=4,hidden_size=32,dropout=0)
    a=fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'a')
    b=fit(tt[:5],tt[5:],['A','B'],replace(cfg,micro_batch=2),'tcn','mean',42,tmp_path/'b')
    for k,v in a['model'].state_dict().items():assert torch.allclose(v,b['model'].state_dict()[k],atol=2e-6)


def test_modern_resume_restores_rng_loader_optimizer_and_best(tmp_path):
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_training import fit
    from cloth_trainer.engine import Cancelled
    torch.set_num_threads(1);tt=trials();cfg=ModernConfig(device='cpu',workers=0,epochs=3,batch_size=2,hidden_size=32)
    direct=fit(tt[:6],tt[6:],['A','B'],cfg,'gru','mean',42,tmp_path/'direct')
    def stop(e):
        if e['type']=='epoch' and e['epoch']==1:raise Cancelled('test interruption')
    with pytest.raises(Cancelled):fit(tt[:6],tt[6:],['A','B'],cfg,'gru','mean',42,tmp_path/'retry',emit=stop)
    events=[];resumed=fit(tt[:6],tt[6:],['A','B'],cfg,'gru','mean',42,tmp_path/'retry',emit=events.append)
    assert [e['epoch'] for e in events if e['type']=='epoch']==[2,3]
    for k,v in direct['model'].state_dict().items():assert torch.equal(v,resumed['model'].state_dict()[k])
    assert direct['best_epoch']==resumed['best_epoch']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='真实CUDA不可用；不以CPU冒充GPU测试')
def test_actual_cuda_forward_backward_and_checkpoint(tmp_path):
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_training import fit
    tt=trials();cfg=ModernConfig(device='cuda',workers=0,epochs=1,batch_size=2,hidden_size=32)
    result=fit(tt[:6],tt[6:],['A','B'],cfg,'gru','mean',42,tmp_path)
    assert next(result['model'].parameters()).is_cuda and result['runtime']['device']=='cuda'


def test_modern_clipping_keeps_large_finite_gradients_and_rejects_nan():
    from cloth_trainer.modern_training import clip
    from cloth_trainer.numerical import NumericalFailure
    m=torch.nn.Linear(2,2);m.weight.grad=torch.full_like(m.weight,1e20);m.bias.grad=torch.zeros_like(m.bias)
    stats=clip(m)
    assert stats['norm']>1e20 and .999<stats['clipped_norm']<=1.000001
    assert torch.isfinite(m.weight.grad).all() and m.weight.grad.abs().sum()>0
    m.weight.grad[0,0]=float('nan')
    with pytest.raises(NumericalFailure):clip(m)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='真实CUDA不可用')
@pytest.mark.parametrize('architecture',['gru','tcn','cera','conv_gru','conv_cera'])
def test_actual_cuda_all_architectures(tmp_path,architecture):
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_training import fit
    tt=trials();cfg=ModernConfig(device='cuda',workers=0,epochs=1,batch_size=2,hidden_size=32)
    r=fit(tt[:6],tt[6:],['A','B'],cfg,architecture,'mean',42,tmp_path/architecture)
    assert next(r['model'].parameters()).is_cuda


def test_probe_oom_after_amp_unscale_retries_with_fresh_state(monkeypatch):
    """CPU control-flow test of AMP state recovery; not a CUDA acceptance test."""
    from cloth_trainer import modern_training as training
    from cloth_trainer.modern_config import ModernConfig
    from cloth_trainer.modern_models import ModernClassifier
    from cloth_trainer.devices import seed_runtime,rng_state
    seed_runtime(42,1);m=ModernClassifier(1,2,'gru','mean',32,0)
    before={k:v.clone() for k,v in m.state_dict().items()};generator=torch.Generator().manual_seed(42)
    original=training.clip;calls=[]
    def exhaust_once(model):
        calls.append(1)
        if len(calls)==1:raise torch.OutOfMemoryError('synthetic optimizer-memory pressure')
        return original(model)
    monkeypatch.setattr(training,'clip',exhaust_once)
    cfg=ModernConfig(device='cpu',workers=0,epochs=1,batch_size=4,hidden_size=32)
    chosen=training.probe_microbatch(m,cfg,torch.device('cpu'),24,1,generator,amp_factory=lambda:torch.amp.GradScaler('cpu'))
    assert chosen==2 and len(calls)==2
    for k,v in m.state_dict().items():assert torch.equal(v,before[k])
