"""Read-only review of current fabric collection and existing development CV."""
import sys,json,csv,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import discover_report,import_dataset,prepare_trials,file_hash
from cloth_trainer.config import DataConfig
from cloth_trainer.quality import RULE_VERSION
DEST=Path(__file__).parent
previous=json.loads((DEST.parent/'2026-11-18-first20/分析结果.json').read_text())
labels=[x['label'] for x in previous['classes']]
files,ignored,errors=discover_report([ROOT.parent/'xz-control/data/布料采集数据'])
fabric=[];other=[]
for p in files:
    m=json.loads((p.parent/'metadata.json').read_text())
    (fabric if m.get('label') in labels else other).append(p)
def progress(i,n):
    if i%50==0 or i==n:print('REVIEW',i,n,flush=True)
records,report=import_dataset(fabric,DataConfig(),signals=('by',),progress=progress)
assert len(records)+len(report['errors'])+len(report['duplicates'])==len(fabric)
rows=[]
for r in records:
    p=Path(r.path);m=json.loads((p.parent/'metadata.json').read_text());d=r.diagnostics['direction_quality']
    assert r.sha256==file_hash(p) and r.metadata_sha256==file_hash(p.parent/'metadata.json')
    ts=prepare_trials([r],'by');amp=[float(np.median(t.values[:,0])) for t in ts]
    rows.append(dict(label=r.label,path=r.path,record=p.stem,sequence=m.get('sequence'),created_at=m.get('created_at'),
       status=r.quality,eligible=r.audit['eligible'],history=r.audit.get('history'),reasons=r.audit['reasons'],
       sha256=r.sha256,metadata_sha256=r.metadata_sha256,group=r.group,
       full_rate=d.get('full_rate'),common_rate=d.get('common_rate'),common_difference=d.get('common_difference'),
       same_sign_fraction=d.get('same_sign_fraction'),marked=d.get('marked'),conditions=d.get('conditions'),
       amplitude=float(np.mean(amp)),out_amplitude=amp[0],back_amplitude=amp[1],baseline_sigma=r.diagnostics['baseline_sigma'][1]))
for e in report['errors']:
    m=json.loads((Path(e['path']).parent/'metadata.json').read_text())
    rows.append(dict(label=m.get('label'),path=e['path'],record=Path(e['path']).stem,sequence=m.get('sequence'),
        status=e['audit']['status'],eligible=False,reasons=[e['message']],sha256=e.get('sha256'),metadata_sha256=e.get('metadata_sha256')))
statusnames={'normal':'审查通过','suggest_review':'可用·建议复核','excluded':'本实验排除','undetermined':'无法判断','data_error':'数据错误'}
classes=[]
for label in labels:
    rr=[r for r in rows if r['label']==label];valid=[r for r in rr if r['eligible']];amps=np.array([r['amplitude'] for r in valid])
    old=next(s for s in previous['classes'] if s['label']==label)
    scores={a:old[a+'_val_f1'] for a in ('gru','tcn','cera')}
    suggestion='优先保留' if min(scores['tcn'],scores['cera'])>=.85 else ('暂不纳入候选' if max(scores['tcn'],scores['cera'])<.80 else '保留待验证')
    confusion={};support={}
    for arch in scores:
        metrics=previous['cv'][arch]['pooled_validation'];idx=labels.index(label);matrix=np.array(metrics['confusion']);arr=matrix[idx].copy();arr[idx]=0
        confusion[arch]=[dict(label=labels[int(i)],count=int(arr[i])) for i in np.argsort(-arr) if arr[i]>0][:3]
        support[arch]=next(x['support'] for x in metrics['per_class'] if x['label']==label)
    bands=[]
    for low,high in ((1,10),(11,20),(21,30),(31,35),(36,45)):
        aa=[r['amplitude'] for r in valid if low<=r['sequence']<=high]
        if aa:bands.append(dict(low=low,high=high,n=len(aa),median=float(np.median(aa)),cv=float(np.std(aa,ddof=1)/np.mean(aa)) if len(aa)>1 else None))
    classes.append(dict(label=label,total=len(rr),eligible=len(valid),excluded=sum(r['status']=='excluded' for r in rr),
        review=sum(r['status']=='suggest_review' for r in rr),unknown=sum(r['status']=='undetermined' for r in rr),data_error=sum(r['status']=='data_error' for r in rr),
        median=float(np.median(amps)),cv=float(np.std(amps,ddof=1)/np.mean(amps)),common_rate_median=float(np.median([r['common_rate'] for r in valid])),
        common_rate_max=float(max(r['common_rate'] for r in valid)),bands=bands,f1=scores,validation_support=support,
        confusion=confusion,suggestion=suggestion,excluded_records=[r['record'] for r in rr if not r['eligible']]))
order={'暂不纳入候选':0,'保留待验证':1,'优先保留':2}
classes.sort(key=lambda s:(order[s['suggestion']],s['f1']['cera']))
result=dict(scope=dict(source=str(ROOT.parent/'xz-control/data/布料采集数据'),fabric_csv=len(fabric),other_csv=len(other),
    other_records=[str(p) for p in other],ignored=len(ignored),review_rule=RULE_VERSION,signal='|ΔBy|',
    model_source=previous['scope']['task'],model_scope='每类前20次；按CSV分组的五折开发集验证；464条方向序列/232个CSV',
    no_new_training=True,formal_test_evaluated=False,source_modified=False),quality=dict(collections.Counter(r['status'] for r in rows)),
    classes=classes,records=rows,batch=report['review_batches'][0]['classes'],duplicates=report['duplicates'],errors=report['errors'],
    cv_summary={a:previous['cv'][a]['fold_mean'] for a in ('gru','tcn','cera')})
(DEST/'评估结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
headers=['布料','当前CSV数','可训练CSV数','排除CSV数','建议复核CSV数','无法判断CSV数','数据错误CSV数','TCN验证F1','CERA验证F1','GRU验证F1','验证方向序列数','可训练幅值中位数μT','同类采集幅值变异系数','去回差异率中位数','筛选建议','CERA主要错分目标','TCN主要错分目标','排除记录']
matrix=[]
for s in classes:
    targets=lambda a:'；'.join(x['label']+'（'+str(x['count'])+'条）' for x in s['confusion'][a]) or '无'
    matrix.append([s['label'],s['total'],s['eligible'],s['excluded'],s['review'],s['unknown'],s['data_error'],s['f1']['tcn'],s['f1']['cera'],s['f1']['gru'],s['validation_support']['cera'],s['median'],s['cv'],s['common_rate_median'],s['suggestion'],targets('cera'),targets('tcn'),'；'.join(s['excluded_records'])])
with (DEST/'布料评估统计.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(headers);w.writerows(matrix)
detailheaders=['布料','采集记录','编号','审查状态','可训练','完整范围差异率','共同范围差异率','共同范围幅值差μT','同向位置箱比例','往返幅值中位数均值μT','处理原因','CSV指纹','metadata指纹','CSV路径']
details=[]
for r in sorted(rows,key=lambda r:(r['label'],r['sequence'] or 0)):
    details.append([r['label'],r['record'],r['sequence'],statusnames.get(r['status'],r['status']),'是' if r['eligible'] else '否',r.get('full_rate'),r.get('common_rate'),r.get('common_difference'),r.get('same_sign_fraction'),r.get('amplitude'),'；'.join(r['reasons']),r.get('sha256'),r.get('metadata_sha256'),r['path']])
with (DEST/'逐次采集审查明细.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(detailheaders);w.writerows(details)
(DEST/'表格输入.json').write_text(json.dumps(dict(headers=headers,rows=matrix,detailheaders=detailheaders,details=details),ensure_ascii=False))
print('SCOPE',result['scope'],flush=True);print('QUALITY',result['quality'],flush=True)
for s in classes:print('CLASS',s['label'],s['total'],s['eligible'],s['excluded'],s['review'],s['f1']['tcn'],s['f1']['cera'],s['suggestion'],s['excluded_records'],flush=True)
print('EXCLUDED',[(r['label'],r['record'],r['reasons']) for r in rows if not r['eligible']],flush=True)
