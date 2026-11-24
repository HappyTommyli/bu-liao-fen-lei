import json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,findfont
root=Path(__file__).resolve().parent
p=root.parents[2]/'outputs/models/CLOTH4_20261120_035813_8ca216'
m=json.loads((p/'metadata.json').read_text())
r=json.loads(next(p.glob('recognition/*.json')).read_text())['results'][0]
new=[x for x in r['predictions'] if not x['training_overlap']]
known=[x for x in new if x['true']>=0];unknown=[x for x in new if x['true']<0]
pairs=collections.defaultdict(list)
for row in known:pairs[row['path']].append(row)
batchids=[]
for s in m['sources']:
 d=json.loads((Path(s['path']).parent/'metadata.json').read_text());batchids.append(d['context'].get('batch',{}).get('id'))
overlap=[]
for f in m['split_plan']['cv']:
 tr={batchids[i] for i in f['train']}
 overlap.append(dict(validation_files=len(f['val']),validation_files_with_training_batch=sum(batchids[i] in tr for i in f['val'])))
cv={}
for key,s in m['summaries'].items():
 folds=[x for x in m['models'].values() if x['configuration_id']==key]
 cv[key]=dict(s,best_epochs=[x['best_epoch'] for x in folds],train_f1=float(np.mean([x['train_metrics']['macro_f1'] for x in folds])),
  best_f1_by_200=float(np.mean([min(x['history'],key=lambda h:abs(h['epoch']-200))['best_f1'] for x in folds])),parameters=folds[0]['parameters'])
result=dict(experiment=m['id'],model=m['final_model']['model_id'],cv=cv,cv_batch_overlap=overlap,
 recognition=dict(accepted_files=len(r['predictions'])//2,rejected_files=len(r['errors']),replay_files=sum(x['training_overlap'] for x in r['predictions'])//2,
 known_new_files=len(known)//2,unknown_new_files=len(unknown)//2,unknown_labels=sorted({x['label'] for x in unknown}),
 known_new_metrics=r['evaluation']['new'],supported_classes_macro_f1=float(np.mean([x['f1'] for x in r['evaluation']['new']['per_class'] if x['support']])),
 known_new_pairs=dict(total=len(pairs),prediction_agreement=sum(v[0]['predicted']==v[1]['predicted'] for v in pairs.values()),
 probability_mean_correct=int(sum(np.argmax(np.mean([x['scores'] for x in v],0))==v[0]['true'] for v in pairs.values()))),
 error_reasons=dict(collections.Counter(x['message'] for x in r['errors']))))
(root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
from matplotlib import font_manager
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[1,1.35]})
means=[s['val_f1_mean']*100 for s in m['summaries'].values()];stds=[s['val_f1_std']*100 for s in m['summaries'].values()]
ax[0].bar(['900 / mean','900 / mean+std','450 / mean','1800 / mean'],means,yerr=stds,color=['#8aa9cf','#3a9a8b','#d8a566','#8aa9cf'],capsize=4)
for i,v in enumerate(means):ax[0].text(i,v-3,f'{v:.2f}',ha='center',va='top',color='white',fontweight='bold')
ax[0].set_ylim(88,102);ax[0].set_ylabel('开发集 Macro-F1（%）');ax[0].set_title('同一数据版本：五折验证均值 ± 折间标准差');ax[0].tick_params(axis='x',rotation=12)
classes=['大衣呢','平纹棉布','桃皮绒','织棉提花缎','岩石肌理面料'];y=np.arange(len(classes));width=.35
for j,d in enumerate(['out','back']):
 vals=[]
 for label in classes:
  rows=[x for x in known if x['label']==label and x['direction']==d];vals.append(sum(x['predicted']==x['true'] for x in rows)/len(rows)*100)
 ax[1].barh(y+(j-.5)*width,vals,height=width,label='去程' if d=='out' else '回程',color=['#507fb0','#d39858'][j])
counts=[sum(x['label']==c for x in known)//2 for c in classes]
ax[1].set_yticks(y,[f'{c}（{n}次采集）' for c,n in zip(classes,counts)]);ax[1].invert_yaxis();ax[1].set_xlim(0,105);ax[1].set_xlabel('该类召回率（%）');ax[1].set_title('独立已知类别新数据：去/回程差异');ax[1].legend(loc='lower right')
fig.suptitle('最新 TCN：同批次验证接近饱和，新批次表现仍不均衡',fontsize=15)
fig.text(.02,.015,'右图仅覆盖 5 / 16 类；岩石肌理面料仅 1 次采集。两图评价范围不同，不作直接性能差值比较。',fontsize=10,color='#555555')
fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(root/'latest-tcn-analysis.png',dpi=160)
print(root/'latest-tcn-analysis.png')
