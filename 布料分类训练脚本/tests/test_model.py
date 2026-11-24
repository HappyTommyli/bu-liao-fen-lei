import numpy as np
import torch
from cloth_trainer.model import CERANet,joint_loss

torch.set_num_threads(2)

def test_future_replacement_does_not_change_prefix():
    torch.manual_seed(1);net=CERANet(1,3).eval();x=torch.randn(2,91,1);changed=x.clone();changed[:,41:]=100*torch.randn_like(changed[:,41:])
    with torch.no_grad():a,_=net(x);b,_=net(changed)
    for k in ['logits','risk_logit','evidence']:
        assert torch.allclose(a[k][:,:41],b[k][:,:41],atol=1e-5,rtol=1e-5)

def test_stream_chunks_match_whole_trial():
    torch.manual_seed(2);net=CERANet(2,3).eval();x=torch.randn(2,87,2)
    with torch.no_grad():
        whole,_=net(x);state=None;pieces=[]
        for a,b in [(0,1),(1,9),(9,42),(42,87)]:
            out,state=net.forward_stream(x[:,a:b],state);pieces.append(out)
    for key in ['logits','risk_logit','evidence','delta']:
        assert torch.allclose(whole[key],torch.cat([p[key] for p in pieces],1),atol=1e-5,rtol=1e-5),key

def test_class_and_risk_losses_both_update_shared_branches():
    torch.manual_seed(3);net=CERANet(1,3).eval();out,_=net(torch.randn(3,60,1));idx=torch.tensor([[5,11,17,29,44,59]]*3)
    total,cls,risk,targets=joint_loss(out,torch.tensor([0,1,2]),idx,torch.ones(6,2))
    assert not targets.requires_grad
    for loss in [cls,risk]:
        grads=torch.autograd.grad(loss,[net.tcn[0].conv.weight,net.gru.weight_ih_l0],retain_graph=True)
        assert all(g.abs().sum()>0 for g in grads)
    total.backward()

