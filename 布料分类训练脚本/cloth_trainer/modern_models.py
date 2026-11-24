"""Whole-segment classifiers, with causal features and masked pooling."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .model import accumulate


class Block(nn.Module):
    def __init__(self,cin,h,dilation,dropout,stride=1):
        super().__init__();self.left=4*dilation;self.stride=stride
        self.conv=nn.Conv1d(cin,h,5,stride=stride,dilation=dilation)
        self.skip=nn.Conv1d(cin,h,1,stride=stride) if cin!=h or stride!=1 else nn.Identity()
        self.norm=nn.LayerNorm(h);self.drop=nn.Dropout(dropout)
    def forward(self,x):
        y=self.conv(F.pad(x.transpose(1,2),(self.left,0))).transpose(1,2)
        skip=self.skip(x.transpose(1,2)).transpose(1,2)
        return self.drop(F.gelu(self.norm(y+skip)))


class ModernClassifier(nn.Module):
    def __init__(self,channels,classes,architecture='gru',readout='mean',hidden=64,dropout=.1):
        super().__init__();self.architecture=architecture;self.readout=readout
        self.compressed=architecture.startswith('conv_');kind=architecture.removeprefix('conv_')
        if kind not in ('gru','tcn','cera') or readout not in ('mean','last'):raise ValueError('模型类型无效')
        self.kind=kind
        self.stem=nn.Sequential(Block(channels,hidden,1,dropout,2),Block(hidden,hidden,1,dropout,2)) if self.compressed else nn.Identity()
        cin=hidden if self.compressed else channels
        if kind in ('gru','cera'):self.gru=nn.GRU(cin,hidden,batch_first=True)
        if kind in ('tcn','cera'):
            self.tcn=nn.Sequential(*[Block(cin if i==0 else hidden,hidden,2**i,dropout) for i in range(6)])
        if kind=='cera':
            self.gate=nn.Linear(2*hidden,hidden);self.add=nn.Linear(hidden,hidden);self.retain=nn.Linear(hidden,hidden)
        self.drop=nn.Dropout(dropout);self.head=nn.Linear(hidden,classes)

    def forward(self,x,lengths):
        x=self.stem(x);lengths=(lengths+3)//4 if self.compressed else lengths
        if self.kind in ('gru','cera'):
            packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            g,_=self.gru(packed);g,_=pad_packed_sequence(g,batch_first=True,total_length=x.shape[1])
        if self.kind in ('tcn','cera'):local=self.tcn(x)
        if self.kind=='gru':z=g
        elif self.kind=='tcn':z=local
        else:
            gate=torch.sigmoid(self.gate(torch.cat([g,local],-1)));z=gate*local+(1-gate)*g
            z=accumulate(torch.sigmoid(self.retain(z)),F.softplus(self.add(z)),z.new_zeros(len(z),z.shape[-1]))
        device_lengths=lengths.to(z.device)
        if self.readout=='mean':
            mask=torch.arange(z.shape[1],device=z.device)[None,:]<device_lengths[:,None]
            feature=(z*mask[:,:,None]).sum(1)/device_lengths[:,None]
        else:feature=z[torch.arange(len(z),device=z.device),device_lengths-1]
        return self.head(self.drop(feature))
