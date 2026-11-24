from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from cloth_trainer.data import Trial
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.tcn_models import default_spec
from cloth_trainer.tcn_study import candidates


def trials():
    rng=np.random.default_rng(12);result=[]
    for i,n in enumerate([301,147,410,128,261,201,333,129]):
        t=np.arange(n)/333
        x=(rng.normal(size=(n,1))*.1+i/5).astype(np.float32)
        result.append(Trial(str(i),f'/toy/{i//2}.csv',f'h{i//2}',f'g{i//2}', 'A' if i<4 else 'B',
                            'out' if i%2==0 else 'back',x,t,t,t,x,(0,1)))
    return result


def config(**kwargs):
    spec=default_spec(32,2);spec['dropout']=0.
    return ModernConfig(device='cpu',workers=0,architectures=('tcn',),tcn_spec=spec,
                        tcn_window=128,epochs=2,batch_size=3,**kwargs)


def test_parent_batch_microbatch_and_tail_weighting_match(tmp_path):
    from cloth_trainer.modern_training import fit,evaluate
    tt=trials();cfg=config()
    a=fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'a',candidate=candidates(cfg)[0])
    b=fit(tt[:5],tt[5:],['A','B'],replace(cfg,micro_batch=1),'tcn','mean',42,tmp_path/'b',candidate=candidates(cfg)[0])
    for key,p in a['model'].state_dict().items():assert torch.allclose(p,b['model'].state_dict()[key],atol=3e-6),key
    assert [r['optimizer_updates'] for r in a['history']]==[2,4]
    assert a['history'][-1]['effective_points']==sum(len(t.values) for t in tt[:5])*2
    rows,loss=evaluate(a['model'],tt,a['scaler'],['A','B'],cfg)
    assert len(rows)==len(tt) and all('windows' in r for r in rows)
    assert [len(r['windows']) for r in rows]==[math.ceil(len(t.values)/128) for t in tt]
    assert math.isfinite(loss)
    for r,t in zip(rows,tt):
        w=r['windows'];expected=sum(np.array(q['scores'])*q['points']/len(t.values) for q in w)
        assert np.allclose(expected,r['scores']) and r['id']==t.id
        assert sum(q['points'] for q in w)==len(t.values)
    assert loss==pytest.approx(np.mean([-math.log(r['scores'][r['true']]) for r in rows]),rel=1e-5)


def test_window_resume_and_changed_candidate_rejected(tmp_path):
    from cloth_trainer.modern_training import fit
    from cloth_trainer.engine import Cancelled
    tt=trials();cfg=config();candidate=candidates(cfg)[0]
    direct=fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'direct',candidate=candidate)
    def stop(e):
        if e['type']=='epoch' and e['epoch']==1:raise Cancelled('interrupted')
    with pytest.raises(Cancelled):fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'resume',emit=stop,candidate=candidate)
    resumed=fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'resume',candidate=candidate)
    for key,p in direct['model'].state_dict().items():assert torch.equal(p,resumed['model'].state_dict()[key])
    with pytest.raises(ValueError,match='校验|关联'):
        fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'resume',candidate=dict(candidate,window_size=256))


def test_custom_clip_threshold_and_unknown_labels(tmp_path):
    from cloth_trainer.modern_training import clip,fit,evaluate
    model=torch.nn.Linear(2,2)
    for p in model.parameters():p.grad=torch.ones_like(p)*10
    stats=clip(model,max_norm=.3)
    assert stats['clipped_norm']==pytest.approx(.3,abs=1e-6)
    tt=trials();cfg=config()
    result=fit(tt[:5],tt[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'a',candidate=candidates(cfg)[0])
    rows,loss=evaluate(result['model'],[replace(tt[0],label='unknown')],result['scaler'],['A','B'],cfg)
    assert len(rows)==1 and rows[0]['true']==-1 and loss is None


def test_window_microbatch_may_exceed_parent_batch(tmp_path):
    from cloth_trainer.modern_training import fit
    cfg=replace(config(),batch_size=2,micro_batch=8)
    assert cfg.validate() is cfg
    result=fit(trials()[:5],trials()[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'a',candidate=candidates(cfg)[0])
    assert result['history'][-1]['optimizer_updates']==6
    with pytest.raises(ValueError):ModernConfig(batch_size=2,micro_batch=8).validate()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='真实CUDA不可用，不以CPU替代')
def test_actual_cuda_window_training_with_large_window_microbatch(tmp_path):
    from cloth_trainer.modern_training import fit
    cfg=replace(config(),device='cuda',batch_size=2,micro_batch=8,epochs=1)
    result=fit(trials()[:5],trials()[5:],['A','B'],cfg,'tcn','mean',42,tmp_path/'gpu',candidate=candidates(cfg)[0])
    assert next(result['model'].parameters()).is_cuda
    assert result['history'][0]['optimizer_updates']==3
    assert result['history'][0]['gpu_peak_memory_bytes']>0
