"""Causal, full-segment classifiers. No risk head or prefix supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import accumulate


class CausalBlock(nn.Module):
    def __init__(self,cin,cout,dilation,dropout):
        super().__init__()
        self.left=4*dilation
        self.conv=nn.Conv1d(cin,cout,5,dilation=dilation)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return F.gelu(self.dropout(self.conv(F.pad(x,(self.left,0))))+self.skip(x))


class SegmentClassifier(nn.Module):
    def __init__(self,channels,classes,architecture='cera',readout='mean',hidden_size=64,dropout=.1):
        super().__init__()
        if architecture not in ('gru','tcn','cera') or readout not in ('mean','last'):
            raise ValueError('未知分类主干或读出方式')
        self.channels=channels;self.architecture=architecture;self.readout=readout
        h=hidden_size
        if architecture in ('gru','cera'):
            self.gru=nn.GRU(channels,h,batch_first=True)
        if architecture in ('tcn','cera'):
            self.tcn=nn.Sequential(CausalBlock(channels,h//2,1,dropout),CausalBlock(h//2,h,2,dropout),CausalBlock(h,h,4,dropout))
        if architecture=='cera':
            self.local_project=nn.Linear(h,h);self.context_project=nn.Linear(h,h)
            self.gate=nn.Linear(2*h,h);self.add=nn.Linear(h,h);self.retain=nn.Linear(h,h)
        self.dropout=nn.Dropout(dropout)
        self.class_head=nn.Linear(h,classes)

    def features(self,x):
        if x.ndim!=3 or x.shape[2]!=self.channels or x.shape[1]<1:
            raise ValueError('输入必须为非空[B,T,C]')
        if self.architecture=='gru':return self.gru(x)[0]
        local=self.tcn(x.transpose(1,2)).transpose(1,2)
        if self.architecture=='tcn':return local
        local=self.local_project(local)
        context=self.context_project(self.gru(x)[0])
        gate=torch.sigmoid(self.gate(torch.cat((local,context),-1)))
        z=gate*local+(1-gate)*context
        return accumulate(torch.sigmoid(self.retain(z)),F.softplus(self.add(z)),x.new_zeros(x.shape[0],z.shape[2]))

    def forward(self,x,lengths):
        lengths=torch.as_tensor(lengths,device=x.device,dtype=torch.long)
        if lengths.shape!=(x.shape[0],) or (lengths<1).any() or (lengths>x.shape[1]).any():
            raise ValueError('有效长度必须覆盖每个非空单程')
        values=self.features(x)
        if self.readout=='mean':
            mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
            pooled=(values*mask[:,:,None]).sum(1)/lengths[:,None]
        else:
            pooled=values[torch.arange(x.shape[0],device=x.device),lengths-1]
        return self.class_head(self.dropout(pooled))
