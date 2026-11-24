"""Read-only inference for preserved cera-net-v1 artifacts."""
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .model import checkpoint_indices,selected
from .config import CHECKPOINTS
from .calibration import stop_trial

def evaluate_legacy(bundle,trials,cancelled):
    model=bundle['_model'];model.eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(trials),8):
            if cancelled():
                from .engine import Cancelled
                raise Cancelled('已取消识别')
            tt=trials[start:start+8]
            values=[torch.from_numpy(((t.values-np.array(bundle['scaler']['mean']))/np.array(bundle['scaler']['std'])).astype(np.float32)) for t in tt]
            x=pad_sequence(values,batch_first=True);state=None;parts=[]
            for a in range(0,x.shape[1],512):
                out,state=model.forward_stream(x[:,a:a+512],state);parts.append(out)
            whole={k:None if parts[0][k] is None else torch.cat([o[k] for o in parts],1) for k in parts[0]}
            out=selected(whole,checkpoint_indices([len(t.values) for t in tt]));prob=out['logits'].softmax(-1).numpy()
            risk=out['risk_logit'].sigmoid().numpy() if out['risk_logit'] is not None else 1-prob.max(-1)
            if bundle['variant'] in ('softmax','no_risk'):risk=1-prob.max(-1)
            elif bundle['variant']=='entropy':risk=-(prob*np.log(np.maximum(prob,1e-12))).sum(-1)/math.log(len(bundle['labels']))
            for i,t in enumerate(tt):
                cps=[dict(checkpoint=q,predicted=int(prob[i,j].argmax()),scores=prob[i,j].tolist(),risk=float(risk[i,j]),
                          evidence=out['evidence'][i,j].tolist(),delta_l1=float(out['delta'][i,j].abs().sum()),elapsed=(math.ceil(len(t.values)*q)-1)/333) for j,q in enumerate(CHECKPOINTS)]
                row=dict(id=t.id,path=t.path,sha256=t.sha256,group=t.group,label=t.label,true=bundle['labels'].index(t.label) if t.label in bundle['labels'] else -1,direction=t.direction,checkpoints=cps)
                row['outcome']=stop_trial(row,bundle['_calibration']);row['predicted']=row['outcome']['predicted'];row['scores']=cps[-1]['scores'];rows.append(row)
    return rows
