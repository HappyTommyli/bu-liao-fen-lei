"""Group exact checkpoints without presenting repetitions as distinct configurations."""
from statistics import mean,pstdev


def configurations(meta):
    grouped={}
    for key,r in meta.get('models',{}).items():
        identity=r.get('configuration_id',key)
        row=grouped.setdefault(identity,dict(id=identity,architecture=r.get('architecture','旧版CERA'),signal=r['signal'],
            readout=r.get('readout','legacy'),candidate=r.get('candidate'),candidate_label=r.get('candidate_label'),keys=[],results=[]))
        row['keys'].append(key);row['results'].append(r)
    for row in grouped.values():
        values=[(r.get('val_metrics') or {}).get('macro_f1') for r in row['results']]
        values=[v for v in values if v is not None]
        row['val_mean']=mean(values) if values else None;row['val_std']=pstdev(values) if values else None
        tests=[r.get('test_metrics',r.get('metrics')) for r in row['results']]
        scores=[v['accuracy'] for v in tests if v is not None]
        row['test_mean']=mean(scores) if len(scores)==len(tests) else None
        row['test_std']=pstdev(scores) if len(scores)==len(tests) else None
        summary=meta.get('summaries',{}).get(row['id'],{}).get('by_seed',{})
        if summary and len(summary)==len({r.get('seed') for r in row['results']}):
            oof=[r['oof_metrics']['accuracy'] for r in summary.values()];row['test_mean']=mean(oof);row['test_std']=pstdev(oof)
    return list(grouped.values())
