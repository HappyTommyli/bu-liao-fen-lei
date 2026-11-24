"""Versioned, configurable causal TCN. Legacy model classes stay untouched."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_REVISION = 'configurable-tcn-v1'


def default_spec(width=64, depth=6):
    return dict(revision=MODEL_REVISION, channels=[width]*depth, kernel_size=5,
                dilations=[2**i for i in range(depth)], dropout=.1, pooling='mean')


def normalize_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError('TCN结构必须为配置对象')
    expected=set(default_spec())
    if set(spec)!=expected or spec.get('revision')!=MODEL_REVISION:
        raise ValueError('TCN结构字段或版本不支持')
    channels,dilations=spec['channels'],spec['dilations']
    integer=lambda v: isinstance(v,int) and not isinstance(v,bool)
    if not isinstance(channels,(list,tuple)) or not 1<=len(channels)<=10 or any(not integer(v) or not 16<=v<=256 for v in channels):
        raise ValueError('TCN通道列表需含1–10项，每项为16–256整数')
    if not isinstance(dilations,(list,tuple)) or len(dilations)!=len(channels) or any(not integer(v) or not 1<=v<=1024 for v in dilations):
        raise ValueError('膨胀系数需与块数一致，每项为1–1024整数')
    kernel=spec['kernel_size']
    if not integer(kernel) or kernel not in range(3,16,2):
        raise ValueError('卷积核需为3–15的奇数')
    dropout=spec['dropout']
    if not isinstance(dropout,(int,float)) or not math.isfinite(dropout) or not 0<=dropout<1:
        raise ValueError('dropout需在[0,1)内')
    if spec['pooling'] not in ('mean','mean_std','max'):
        raise ValueError('特征汇总方式无效')
    return dict(revision=MODEL_REVISION,channels=list(channels),kernel_size=kernel,
                dilations=list(dilations),dropout=float(dropout),pooling=spec['pooling'])


class TCNBlock(nn.Module):
    def __init__(self,cin,cout,kernel,dilation,dropout):
        super().__init__()
        self.left=(kernel-1)*dilation
        self.conv=nn.Conv1d(cin,cout,kernel,dilation=dilation)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
        self.norm=nn.LayerNorm(cout)
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        y=self.conv(F.pad(x.transpose(1,2),(self.left,0))).transpose(1,2)
        skip=self.skip(x.transpose(1,2)).transpose(1,2)
        return self.drop(F.gelu(self.norm(y+skip)))


class TCNClassifier(nn.Module):
    def __init__(self,channels,classes,spec):
        super().__init__()
        self.spec=normalize_spec(spec)
        blocks=[];cin=channels
        for cout,dilation in zip(self.spec['channels'],self.spec['dilations']):
            blocks.append(TCNBlock(cin,cout,self.spec['kernel_size'],dilation,self.spec['dropout']))
            cin=cout
        self.blocks=nn.Sequential(*blocks)
        self.drop=nn.Dropout(self.spec['dropout'])
        self.head=nn.Linear(cin*(2 if self.spec['pooling']=='mean_std' else 1),classes)
        self.receptive_field=1+(self.spec['kernel_size']-1)*sum(self.spec['dilations'])

    def features(self,x):
        return self.blocks(x)

    def forward(self,x,lengths):
        lengths=torch.as_tensor(lengths,device=x.device,dtype=torch.long)
        if x.ndim!=3 or lengths.shape!=(len(x),) or bool(((lengths<1)|(lengths>x.shape[1])).any()):
            raise ValueError('TCN输入与有效长度不匹配')
        z=self.features(x)
        mask=torch.arange(z.shape[1],device=z.device)[None,:]<lengths[:,None]
        if self.spec['pooling']=='max':
            pooled=z.masked_fill(~mask[:,:,None],-torch.inf).max(1).values
        else:
            mean=z.masked_fill(~mask[:,:,None],0).sum(1)/lengths[:,None]
            if self.spec['pooling']=='mean_std':
                variance=((z-mean[:,None,:]).square().masked_fill(~mask[:,:,None],0)).sum(1)/lengths[:,None]
                # clamp before sqrt prevents an infinite sqrt gradient at variance=0.
                std=variance.clamp_min(1e-12).sqrt().masked_fill(variance==0,0)
                pooled=torch.cat((mean,std),-1)
            else:pooled=mean
        return self.head(self.drop(pooled))
