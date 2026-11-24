"""Parallel causal TCN/GRU with gated evidence recurrence and joint heads."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class CausalBlock(nn.Module):
    def __init__(self,cin,cout,dilation):
        super().__init__();self.left=4*dilation
        self.conv=nn.Conv1d(cin,cout,5,dilation=dilation)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
        self.dropout=nn.Dropout(.1)
    def forward(self,x):
        return F.gelu(self.dropout(self.conv(F.pad(x,(self.left,0))))+self.skip(x))

def accumulate(beta,added,initial):
    """Associative affine prefix scan, algebraically e[t]=beta[t]*e[t-1]+a[t].

    No division/cumulative-product normalization: stays stable on long scans.
    Each doubling uses only previous timesteps and retains full gradients.
    """
    a,b=beta,added
    offset=1
    while offset<b.shape[1]:
        b=torch.cat((b[:,:offset],b[:,offset:]+a[:,offset:]*b[:,:-offset]),1)
        a=torch.cat((a[:,:offset],a[:,offset:]*a[:,:-offset]),1)
        offset*=2
    return b+a*initial[:,None,:]

class CERANet(nn.Module):
    def __init__(self,channels,classes,variant='full'):
        super().__init__();self.channels=channels;self.classes=classes;self.variant=variant
        self.tcn=nn.Sequential(CausalBlock(channels,32,1),CausalBlock(32,64,2),CausalBlock(64,64,4))
        self.gru=nn.GRU(channels,64,batch_first=True)
        self.local_project=nn.Linear(64,64);self.context_project=nn.Linear(64,64)
        self.gate=nn.Linear(128,64);self.plain=nn.Linear(128,64)
        self.add=nn.Linear(64,64);self.retain=nn.Linear(64,64)
        self.class_head=nn.Linear(64,classes)
        self.risk_head=None if variant=='no_risk' else nn.Sequential(nn.Linear(67,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,x,state=None):
        if x.ndim!=3 or x.shape[2]!=self.channels or x.shape[1]==0:raise ValueError('输入形状必须为[B,T,C]')
        state=state or {};cache=state.get('cache')
        past=x if cache is None else torch.cat((cache,x),1)
        local=self.local_project(self.tcn(past.transpose(1,2)).transpose(1,2)[:,-x.shape[1]:])
        context,h=self.gru(x,state.get('gru'));context=self.context_project(context)
        if self.variant=='local':z=local
        elif self.variant=='context':z=context
        elif self.variant=='plain':z=self.plain(torch.cat((local,context),-1))
        else:
            gate=torch.sigmoid(self.gate(torch.cat((local,context),-1)))
            z=gate*local+(1-gate)*context
        initial=state.get('evidence',x.new_zeros(x.shape[0],64))
        if self.variant=='no_evidence':e=z
        else:e=accumulate(torch.sigmoid(self.retain(z)),F.softplus(self.add(z)),initial)
        delta=e-torch.cat((initial[:,None,:],e[:,:-1]),1)
        logits=self.class_head(e);p=logits.softmax(-1)
        entropy=-(p*p.clamp_min(1e-12).log()).sum(-1,keepdim=True)
        top=p.topk(2,dim=-1).values;margin=top[...,:1]-top[...,1:2]
        features=torch.cat((e,entropy,margin,delta.abs().sum(-1,keepdim=True)),-1)
        risk=self.risk_head(features).squeeze(-1) if self.risk_head is not None else None
        outputs=dict(logits=logits,risk_logit=risk,evidence=e,delta=delta)
        return outputs,dict(cache=past[:,-28:],gru=h,evidence=e[:,-1])

    def forward_stream(self,x,state=None):
        if self.training:raise ValueError('流式推理需要eval模式')
        return self.forward(x,state)

def checkpoint_indices(lengths):
    from .config import CHECKPOINTS
    return torch.tensor([[max(0,math.ceil(n*q)-1) for q in CHECKPOINTS] for n in lengths],dtype=torch.long)

def selected(outputs,indices):
    b=torch.arange(len(indices),device=indices.device)[:,None]
    return {k:(None if v is None else v[b,indices]) for k,v in outputs.items()}

def joint_loss(outputs,targets,indices,weights):
    out=selected(outputs,indices);logits=out['logits'];truth=targets[:,None].expand(-1,6)
    cls=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1))
    errors=(logits.detach().argmax(-1)!=truth).to(logits.dtype)
    if out['risk_logit'] is None:risk=cls*0.
    else:
        w=weights[None,:,0]*(1-errors)+weights[None,:,1]*errors
        risk=(F.binary_cross_entropy_with_logits(out['risk_logit'],errors,reduction='none')*w).mean()
    return cls+risk,cls,risk,errors
