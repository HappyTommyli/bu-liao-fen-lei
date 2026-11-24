"""Read-only frozen-model audit. Writes derived analysis only; never trains."""
import json,sys,collections,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/serein/classify-cloth/布料分类训练脚本')
sys.path.insert(0,str(ROOT))
from cloth_trainer.data import load_record,file_hash,metadata_hash,prepare_trials
from cloth_trainer.config import TrainConfig
from cloth_trainer.engine import load_bundle,evaluate,classification_metrics
RUN=ROOT/'outputs/models/CERA_20261114_171001_3b31d8'
OUT=Path(__file__).parent
meta=json.loads((RUN/'metadata.json').read_text());cfg=TrainConfig.from_dict(meta['config'])
torch.set_num_threads(2)
result={'run':str(RUN),'status':meta['status'],'models':{},'source_integrity':{},'partition_audit':{}}
records=[]
for source in meta['sources']:
    assert file_hash(source['path'])==source['sha256'],source['path']+' CSV changed'
    assert metadata_hash(source['path'])==source['metadata_sha256'],source['path']+' metadata changed'
    r=load_record(source['path'],cfg.data,source['label']);assert r.group==source['group'];records.append(r)
print('Verified original CSV and metadata:',len(records),flush=True)
result['source_integrity']={'csv_verified':len(records),'metadata_verified':len(records),'unique_csv':len({r.sha256 for r in records})}
sets={k:[records[i] for i in ix] for k,ix in meta['partitions'].items()}
assert sorted(i for ix in meta['partitions'].values() for i in ix)==list(range(len(records)))
for k,rs in sets.items():
    result['partition_audit'][k]={'csv':len(rs),'trials':len(rs)*2,'groups':len({r.group for r in rs}),'labels':dict(collections.Counter(r.label for r in rs))}
    for j,ss in sets.items():
        if j!=k:assert not {r.group for r in rs}&{r.group for r in ss},f'group leakage {k} {j}'
print('Partition groups disjoint',flush=True)
# Raw input magnitudes/direction, summarized without changing data.
raw=[]
for r in records:
 for t in r.trials:
    windows=[]
    for a,b in [(0,.1),(.45,.55),(.9,1.)]:
        v=t.values[int(len(t.values)*a):int(len(t.values)*b)]
        windows.append({'q':[a,b],'mean':v.mean(0).tolist(),'std':v.std(0).tolist(),'rms':np.sqrt((v*v).mean(0)).tolist()})
    norm=np.linalg.norm(t.values,axis=1);absy=np.abs(t.values[:,1]);rms=np.sqrt((t.values**2).mean(0))
    raw.append({'id':t.id,'label':t.label,'direction':t.direction,'windows':windows,'rms':rms.tolist(),'x_rms_fraction':float(rms[0]/np.linalg.norm(rms)),
                'by_negative_fraction':float((t.values[:,1]<0).mean()),'norm_absy_correlation':float(np.corrcoef(norm,absy)[0,1])})
result['input_statistics']=raw
for key in ['by_full','bxby_full','xy_full']:
 b=load_bundle(RUN/key/'model.pt');model=b['_model'];r=json.loads((RUN/key/'result.json').read_text());out={'partitions':{},'scaler':b['scaler'],'parameters':sum(p.numel() for p in model.parameters())}
 for name,rs in sets.items():
    rows=evaluate(model,prepare_trials(rs,b['signal']),b['scaler'],b['labels'],stream=True)
    metrics=[classification_metrics(rows,b['labels'],i) for i in range(6)]
    out['partitions'][name]={'metrics':metrics,'rows':[{kk:vv for kk,vv in row.items() if kk!='checkpoints'}|{'checkpoints':[{kk:vv for kk,vv in cp.items() if kk!='evidence'} for cp in row['checkpoints']]} for row in rows]}
    if name=='val':
        best=next(h for h in r['history'] if h['epoch']==r['best_epoch'])
        assert abs(np.mean([m['macro_f1'] for m in metrics])-best['val_prefix_f1'])<1e-8
    if name=='test':
        saved={row['id']:row for row in r['test_predictions']}
        assert set(saved)=={row['id'] for row in rows}
        delta=max(abs(np.array(cp['scores'])-np.array(saved[row['id']]['checkpoints'][i]['scores'])).max() for row in rows for i,cp in enumerate(row['checkpoints']))
        assert delta<1e-5,delta
        out['test_replay_max_probability_delta']=float(delta)
        for i,m in enumerate(metrics):assert m['confusion']==r['prefix_metrics'][i]['confusion']
    print(key,name,'accuracy',round(metrics[-1]['accuracy'],4),'mean_prefix_f1',round(np.mean([m['macro_f1'] for m in metrics]),4),flush=True)
 result['models'][key]=out
 (OUT/'冻结模型复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('Saved',OUT/'冻结模型复核.json',flush=True)
