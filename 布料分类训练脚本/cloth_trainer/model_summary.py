"""Exact cheap parameter counts, without allocating neural networks on UI changes."""
from functools import lru_cache
import json
from .cera_models import normalize_model_spec,CERA_REVISIONS


def parameter_summary(spec,classes,inputs=1):
    return dict(_summary(json.dumps(normalize_model_spec(spec),sort_keys=True),classes,inputs))


@lru_cache(maxsize=128)
def _summary(encoded,classes,inputs):
    spec=json.loads(encoded);cin=inputs;total=0
    for cout in spec['channels']:
        total+=cin*cout*spec['kernel_size']+cout+2*cout
        if cin!=cout:total+=cin*cout+cout
        cin=cout
    total+=cin*(2 if spec['pooling']=='mean_std' else 1)*classes+classes
    if spec['revision'] in CERA_REVISIONS:
        hidden=spec['gru_hidden'];incoming=inputs
        for _ in range(spec['gru_layers']):total+=3*hidden*incoming+3*hidden*hidden+6*hidden;incoming=hidden
        if hidden!=cin:total+=hidden*cin+cin
        total+=2*cin*cin+cin+2*(cin*cin+cin)
    return dict(parameters=total,receptive_field=1+(spec['kernel_size']-1)*sum(spec['dilations']))
