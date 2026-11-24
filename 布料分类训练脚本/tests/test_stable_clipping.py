import math
import pytest
import torch
from cloth_trainer import numerical


def parameter(gradient):
    p=torch.nn.Parameter(torch.ones_like(gradient));p.grad=gradient.clone();return p


def test_huge_finite_gradients_keep_direction_and_nonzero_updates():
    p=parameter(torch.tensor([3e30,4e30]));q=parameter(torch.tensor([-12e30]))
    stats=numerical.clip_gradients([('p',p),('q',q)])
    assert math.isfinite(stats['norm']) and stats['norm']==pytest.approx(13e30,rel=1e-6)
    assert stats['clipped'] and stats['coefficient']>0 and stats['max_abs']==pytest.approx(12e30,rel=1e-6)
    assert torch.allclose(p.grad,torch.tensor([3/13,4/13])) and torch.allclose(q.grad,torch.tensor([-12/13]))
    assert stats['clipped_norm']==pytest.approx(1.,abs=1e-6)
    optimizer=torch.optim.AdamW([p,q],lr=.001);optimizer.step()
    assert torch.isfinite(p).all() and torch.isfinite(q).all()
    assert not torch.equal(p,torch.ones_like(p))


@pytest.mark.parametrize('gradient',[torch.zeros(2),torch.tensor([.3,.4]),torch.tensor([3.,4.])])
def test_zero_and_ordinary_gradients_are_clipped_only_when_needed(gradient):
    p=parameter(gradient);stats=numerical.clip_gradients([('p',p)])
    expected=gradient if gradient.norm()<=1 else gradient/gradient.norm()
    assert torch.allclose(p.grad,expected,atol=1e-6)
    assert stats['clipped']==bool(gradient.norm()>1) and stats['clipped_norm']<=1+1e-6


@pytest.mark.parametrize('bad',[float('inf'),float('-inf'),float('nan')])
def test_invalid_gradient_rejected_before_mutating_any_gradient(bad):
    p=parameter(torch.tensor([3.,4.]));q=parameter(torch.tensor([bad]))
    with pytest.raises(numerical.NumericalFailure,match='非有限'):numerical.clip_gradients([('p',p),('q',q)])
    assert torch.equal(p.grad,torch.tensor([3.,4.]))


def test_max_float32_gradients_have_a_finite_norm_and_nonzero_clip():
    p=parameter(torch.full((64,),torch.finfo(torch.float32).max))
    stats=numerical.clip_gradients([('p',p)])
    assert math.isfinite(stats['norm']) and torch.allclose(p.grad,torch.full((64,),.125),atol=1e-6)
