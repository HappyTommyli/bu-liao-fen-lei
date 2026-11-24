"""Window CERA: parallel causal GRU/TCN, gated evidence accumulation.

Every forward starts from zero state; historical whole-segment CERA is unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .tcn_models import TCNClassifier,default_spec,normalize_spec
from .model import accumulate

MODEL_REVISION='configurable-cera-v1'
PREFIX_REVISION='configurable-cera-prefix-v1'
CERA_REVISIONS=(MODEL_REVISION,PREFIX_REVISION)


def default_cera_spec(width=128,depth=6,gru_hidden=128,gru_layers=1):
    return dict(default_spec(width,depth),revision=MODEL_REVISION,gru_hidden=gru_hidden,gru_layers=gru_layers)


def normalize_cera_spec(spec):
    if not isinstance(spec,dict) or set(spec)!=set(default_cera_spec()) or spec.get('revision')!=MODEL_REVISION:
        raise ValueError('CERA结构字段或版本不支持')
    hidden,layers=spec['gru_hidden'],spec['gru_layers']
    if type(hidden) is not int or not 16<=hidden<=256 or type(layers) is not int or not 1<=layers<=4:
        raise ValueError('GRU隐藏维度需为16–256，层数需为1–4')
    common={k:v for k,v in spec.items() if k not in ('gru_hidden','gru_layers')}
    common['revision']=default_spec()['revision'];common=normalize_spec(common)
    if common['pooling']!='mean':raise ValueError('当前CERA版本使用均值汇总')
    return dict(common,revision=MODEL_REVISION,gru_hidden=hidden,gru_layers=layers)


def normalize_model_spec(spec):
    if isinstance(spec,dict) and spec.get('revision') in CERA_REVISIONS:
        return dict(normalize_cera_spec(dict(spec,revision=MODEL_REVISION)),revision=spec['revision'])
    return normalize_spec(spec)


class CERAClassifier(TCNClassifier):
    def __init__(self,channels,classes,spec):
        normalized=normalize_model_spec(spec)
        common={k:v for k,v in normalized.items() if k not in ('gru_hidden','gru_layers')}
        common['revision']=default_spec()['revision']
        super().__init__(channels,classes,common);self.spec=normalized
        hidden=normalized['gru_hidden'];layers=normalized['gru_layers'];width=normalized['channels'][-1]
        self.gru=nn.GRU(channels,hidden,num_layers=layers,batch_first=True,dropout=normalized['dropout'] if layers>1 else 0.)
        self.context_project=nn.Linear(hidden,width) if hidden!=width else nn.Identity()
        self.gate=nn.Linear(2*width,width);self.add=nn.Linear(width,width);self.retain=nn.Linear(width,width)

    def features(self,x):
        local=self.blocks(x)
        context,_=self.gru(x);context=self.context_project(context)
        gate=torch.sigmoid(self.gate(torch.cat((context,local),-1)))
        fused=gate*local+(1-gate)*context
        return accumulate(torch.sigmoid(self.retain(fused)),F.softplus(self.add(fused)),fused.new_zeros(len(fused),fused.shape[-1]))
