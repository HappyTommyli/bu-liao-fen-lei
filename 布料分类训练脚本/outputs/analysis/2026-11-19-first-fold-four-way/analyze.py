import json,collections,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'outputs/models/CLOTH4_20261119_171027_2dfe99'
OUT=Path(__file__).parent
m=json.loads((SRC/'metadata.json').read_text())
ids=['full_h64','full_h128','w900_h64','w900_h128']
names=['完整单程＋64通道','完整单程＋128通道','900点窗口＋64通道','900点窗口＋128通道']
plt.rcParams['font.sans-serif']=['Noto Sans CJK JP','WenQuanYi Zen Hei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
results=[]
for key in ids:
 p=SRC/f'tcn_by_mean_{key}_s42_f1'
 if not (p/'result.json').exists():
  print('PENDING',key,flush=True);sys.exit(2)
 r=json.loads((p/'result.json').read_text());assert r['status']=='completed' and r['fold']==0
 results.append(r)
assert all([x['id'] for x in r['val_predictions']]==[x['id'] for x in results[0]['val_predictions']] for r in results)
parts=m['partitions'][0]
groups={key:{m['sources'][i]['group'] for i in indices} for key,indices in dict(train=parts['train'],val=parts['val'],test=m['split_plan']['test']).items()}
assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
summary=[]
for r in results:
 h=r['history'];v=r['val_metrics']
 row=dict(id=r['candidate']['id'],status=r['status'],parameters=r['parameters'],epochs=len(h),best_epoch=r['best_epoch'],train_accuracy=r['train_metrics']['accuracy'],accuracy=v['accuracy'],macro_f1=v['macro_f1'],correct=v['correct'],total=v['total'],val_loss=r['val_loss'],training_seconds=r['training_seconds'],peak_allocated_MiB=max(x['gpu_peak_memory_bytes'] for x in h)/1024**2,direction={k:dict(accuracy=v['accuracy'],correct=v['correct'],total=v['total']) for k,v in r['val_direction_metrics'].items()},window_accuracy=r['val_window_metrics']['accuracy'],updates=h[-1]['optimizer_updates'],effective_points=h[-1]['effective_points'],test_status=r['test_status'])
 row['errors']=[dict(true=m['labels'][i],predicted=m['labels'][j],count=n) for i,line in enumerate(v['confusion']) for j,n in enumerate(line) if i!=j and n]
 summary.append(row)
 print(json.dumps(row,ensure_ascii=False),flush=True)
fig,axes=plt.subplots(1,2,figsize=(13,4.6))
colors=['#4778ae','#df8b2d','#3f9e80','#a35ca3']
for r,name,c in zip(results,names,colors):
 h=r['history'];x=[v['optimizer_updates'] for v in h]
 axes[0].plot(x,[v['val_f1']*100 for v in h],label=name,color=c,lw=1.25)
 axes[1].plot(x,[v['val_loss'] for v in h],label=name,color=c,lw=1.25)
for ax in axes:ax.set_xlabel('累计参数更新次数');ax.grid(alpha=.2)
axes[0].set_ylabel('验证集宏平均F1（%）');axes[1].set_ylabel('验证集损失（单程NLL）');axes[0].legend(fontsize=9)
fig.suptitle('第一折四组对照｜相同54次采集、108个验证单程');fig.tight_layout();fig.savefig(OUT/'第一折训练曲线.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(19,17))
for ax,r,name in zip(axes.flat,results,names):
 cm=np.array(r['val_metrics']['confusion']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=8)
 for i in range(len(cm)):
  for j in range(len(cm)):
   if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>=5 else '#333')
 ax.set_xticks(range(len(cm)),m['labels'],rotation=70,ha='right',fontsize=9)
 ax.set_yticks(range(len(cm)),m['labels'],fontsize=9)
 ax.set_xlabel('预测布料');ax.set_ylabel('真实布料');ax.set_title(f'{name}｜正确{r["val_metrics"]["correct"]}/108',fontsize=13)
fig.tight_layout();fig.savefig(OUT/'第一折四组混淆矩阵.png',dpi=140);plt.close(fig)
lines=['# 第一折四组对照分析','',f'来源：{SRC}。实验：{m["name"]}。仅分析已冻结验证预测，未重新训练或使用最终测试集。','', '17类、330次采集；本折209次训练（418单程）、54次验证（108单程）；另67次为固定测试集。去回程同组，无组交叉。输入|ΔBy|，GPU训练。此实验使用旧固定X窗，样本序号001～020。','', '| 配置 | 训练准确率 | 验证准确率 | 宏F1 | 正确数 | 最佳轮次 | 训练秒数 | 峰值已分配显存MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in zip(names,summary):lines.append(f'|{name}|{r["train_accuracy"]:.2%}|{r["accuracy"]:.2%}|{r["macro_f1"]:.2%}|{r["correct"]}/108|{r["best_epoch"]}|{r["training_seconds"]:.1f}|{r["peak_allocated_MiB"]:.1f}|')
lines+=['','宏F1对每种布料等权，准确率按全部单程计数。表中为按验证宏F1选择的最佳权重，并非最后一轮权重。峰值显存是PyTorch已分配量，不含全部CUDA进程占用。','', '| 布料 | 完整64 | 完整128 | 窗口64 | 窗口128 |','|---|---:|---:|---:|---:|']
for i,label in enumerate(m['labels']):
 vals=[f'{r["val_metrics"]["confusion"][i][i]}/{r["val_metrics"]["per_class"][i]["support"]}' for r in results]
 lines.append('|'+label+'|'+'|'.join(vals)+'|')
lines+=['','上述各布料单元格为正确单程数/验证单程数。','']
for name,r in zip(names,summary):
 lines += ['## '+name,'',f'去程准确率{r["direction"]["out"]["accuracy"]:.2%}；回程准确率{r["direction"]["back"]["accuracy"]:.2%}。','']
 lines += [f'- {e["true"]} → {e["predicted"]}：{e["count"]}个单程' for e in r['errors']]
lines += ['','这里只是一折验证，不能替代五折稳定性和独立测试结论。各类仅3～4次独立验证采集，去回两个单程相关；不据此排除布料或判定原始数据质量。900点方案将约16秒单程切成6个片段，按片段长度加权汇总概率；不等同于只扫900点即可实现表中准确率。','']
(OUT/'第一折四组分析.md').write_text('\n'.join(lines))
(OUT/'第一折结果摘要.json').write_text(json.dumps(dict(experiment=str(SRC),summary=summary),ensure_ascii=False,indent=2))
