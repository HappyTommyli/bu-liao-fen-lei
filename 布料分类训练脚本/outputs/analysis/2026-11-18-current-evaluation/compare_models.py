import json,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.evaluation import classification_metrics
from cloth_trainer.data import file_hash
DEST=Path(__file__).parent
d=json.loads((DEST/'评估结果.json').read_text())
task=ROOT/'outputs/models/CLOTH4_20261117_170039_86f788'
m=json.loads((task/'metadata.json').read_text());snapshot=json.loads((task/'dataset/dataset.json').read_text())
assert len(snapshot['sources'])==430
for e in snapshot['sources']:
    assert file_hash(e['path'])==e['sha256'] and file_hash(Path(e['path']).parent/'metadata.json')==e['metadata_sha256']
cv={}
for arch in ('gru','tcn','cera'):
    predictions=[r for model in m['models'].values() if model['architecture']==arch and model['signal']=='by' and model['readout']=='mean' for r in model['val_predictions']]
    assert len(predictions)==len({r['id'] for r in predictions})==686
    cv[arch]=classification_metrics(predictions,m['labels'])
for s in d['classes']:
    label=s['label'];s['larger_csv']=sum(e['label']==label for e in snapshot['sources']);s['larger_f1']={};s['larger_confusion']={}
    for arch,metrics in cv.items():
        ss=next(p for p in metrics['per_class'] if p['label']==label);s['larger_f1'][arch]=ss['f1'];s['larger_support']=ss['support']
        idx=m['labels'].index(label);counts=metrics['confusion'][idx]
        s['larger_confusion'][arch]=sorted([dict(label=c,count=counts[i]) for i,c in enumerate(m['labels']) if c!=label and counts[i]],key=lambda r:-r['count'])[:3]
    if max(s['f1']['tcn'],s['f1']['cera'],s['larger_f1']['tcn'],s['larger_f1']['cera'])<.8:
        suggestion='优先暂不纳入'
    elif max(s['larger_f1']['tcn'],s['larger_f1']['cera'])<.8:
        suggestion='重点复核'
    elif min(s['f1']['tcn'],s['f1']['cera'],s['larger_f1']['tcn'],s['larger_f1']['cera'])>=.85:
        suggestion='优先保留'
    else:suggestion='保留待验证'
    s['suggestion']=suggestion
rank={'优先暂不纳入':0,'重点复核':1,'保留待验证':2,'优先保留':3}
d['classes'].sort(key=lambda s:(rank[s['suggestion']],max(s['larger_f1']['tcn'],s['larger_f1']['cera'])))
d['larger_model']=dict(task=str(task),source_csv=430,validation_sequences=686,validation_csv=343,
    counts={s['label']:s['larger_csv'] for s in d['classes']},cv=cv)
(DEST/'评估结果.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
headers=['布料','当前CSV','可训练CSV','不可训练CSV','建议复核CSV','前20实验TCN F1','前20实验CERA F1','430实验TCN F1','430实验CERA F1','430实验该类CSV数','430实验验证方向数','共同范围去回差异率中位数','同类采集幅值变异系数','筛选建议','430实验TCN主要错分目标','430实验CERA主要错分目标']
rows=[]
for s in d['classes']:
    text=lambda a:'；'.join(x['label']+'（'+str(x['count'])+'条）' for x in s['larger_confusion'][a]) or '无'
    rows.append([s['label'],s['total'],s['eligible'],s['total']-s['eligible'],s['review'],s['f1']['tcn'],s['f1']['cera'],s['larger_f1']['tcn'],s['larger_f1']['cera'],s['larger_csv'],s['larger_support'],s['common_rate_median'],s['cv'],s['suggestion'],text('tcn'),text('cera')])
with (DEST/'布料评估统计.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(headers);w.writerows(rows)
print('GROUPS',[(s['label'],s['suggestion'],s['larger_csv'],s['larger_support'],s['larger_f1']) for s in d['classes']])
