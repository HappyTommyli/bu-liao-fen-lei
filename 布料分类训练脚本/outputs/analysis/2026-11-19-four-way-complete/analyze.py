import json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
SRC=ROOT/'outputs/models/CLOTH4_20261119_171027_2dfe99'
m=json.loads((SRC/'metadata.json').read_text())
ids=['full_h64','full_h128','w900_h64','w900_h128'];names=['完整64','完整128','窗口900＋64','窗口900＋128']
allrows=[];summaries=[];curves=[]
for key,name in zip(ids,names):
 rr=[json.loads((SRC/f'tcn_by_mean_{key}_s42_f{fold}/result.json').read_text()) for fold in range(1,6)]
 rows=[]
 for r in rr:
  h=r['history'];minloss=float('inf');stale=0;maxstale=0
  for v in h:
   stale=0 if v['val_loss']<minloss-1e-4 else stale+1;minloss=min(minloss,v['val_loss']);maxstale=max(stale,maxstale)
  best=lambda n:max(v['val_f1'] for v in h[:n])
  x=dict(id=key,fold=r['fold']+1,epochs=len(h),best_epoch=r['best_epoch'],f1=r['val_metrics']['macro_f1'],accuracy=r['val_metrics']['accuracy'],train_accuracy=r['train_metrics']['accuracy'],best150=best(150),best180=best(180),best200=best(200),f1gain150_200=best(200)-best(150),val_loss150=h[149]['val_loss'],val_loss200=h[-1]['val_loss'],last50_loss_slope=float(np.polyfit(np.arange(50),[v['val_loss'] for v in h[-50:]],1)[0]),final_lr=h[-1]['lr'],lr_changes=[(v['epoch'],v['lr']) for j,v in enumerate(h) if j and v['lr']!=h[j-1]['lr']],stale=stale,max_stale=maxstale,training_seconds=r['training_seconds'],parameters=r['parameters'],correct=r['val_metrics']['correct'],total=r['val_metrics']['total'],out=r['val_direction_metrics']['out']['accuracy'],back=r['val_direction_metrics']['back']['accuracy'],last20_f1mean=float(np.mean([v['val_f1'] for v in h[-20:]])))
  rows.append(x)
 allrows.extend(rows)
 cm=np.sum([r['val_metrics']['confusion'] for r in rr],axis=0);assert cm.sum()==526
 pc=[dict(label=label,support=int(cm[i].sum()),correct=int(cm[i,i]),recall=float(cm[i,i]/cm[i].sum()))for i,label in enumerate(m['labels'])]
 s=dict(id=key,name=name,f1_mean=float(np.mean([r['f1'] for r in rows])),f1_std=float(np.std([r['f1'] for r in rows],ddof=1)),accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),oof_correct=int(cm.trace()),oof_total=int(cm.sum()),oof_accuracy=float(cm.trace()/cm.sum()),train_accuracy=float(np.mean([r['train_accuracy']for r in rows])),best_epoch=[r['best_epoch']for r in rows],f1gain150_200_mean=float(np.mean([r['f1gain150_200']for r in rows])),lr_reduced_folds=sum(bool(r['lr_changes'])for r in rows),max_stale=max(r['max_stale']for r in rows),last50_loss_slope_mean=float(np.mean([r['last50_loss_slope']for r in rows])),val_loss150_mean=float(np.mean([r['val_loss150']for r in rows])),val_loss200_mean=float(np.mean([r['val_loss200']for r in rows])),training_minutes=sum(r['training_seconds']for r in rows)/60,per_class=pc,rows=rows)
 summaries.append(s)
 curves.append(dict(f1=np.array([[v['val_f1'] for v in r['history']]for r in rr]),loss=np.array([[v['val_loss']for v in r['history']]for r in rr]),train=np.array([[v['loss']for v in r['history']]for r in rr])))
 print(json.dumps({k:v for k,v in s.items()if k not in ('rows','per_class')},ensure_ascii=False),flush=True)
 for row in rows:print('FOLD',json.dumps(row,ensure_ascii=False),flush=True)
print('BEST_EPOCH_GE180',sum(r['best_epoch']>=180 for r in allrows),'OF',len(allrows))
print('WEAK_CLASSES_WINDOW128',sorted(summaries[-1]['per_class'],key=lambda x:x['recall'])[:7])
plt.rcParams['font.sans-serif']=['Noto Sans CJK JP','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,3,figsize=(16,4.6));colors=['#4778ae','#df8b2d','#3f9e80','#a35ca3'];xx=np.arange(1,201)
for name,c,d in zip(names,colors,curves):
 for ax,metric,factor in zip(axes,['f1','loss','train'],[100,1,1]):
  a=d[metric]*factor;mean=a.mean(0);sd=a.std(0,ddof=1)
  ax.plot(xx,mean,color=c,label=name,lw=1.5);ax.fill_between(xx,mean-sd,mean+sd,color=c,alpha=.10)
for ax,title in zip(axes,['验证宏F1（%）','验证损失（单程NLL）','训练损失（片段加权CE）']):
 ax.set_title(title);ax.set_xlabel('训练轮次');ax.axvline(150,color='grey',ls='--',alpha=.5);ax.grid(alpha=.2)
axes[0].legend(fontsize=9);fig.suptitle('四组完整五折｜实线为五折均值，阴影为折间标准差');fig.tight_layout();fig.savefig(OUT/'五折训练曲线.png',dpi=160);plt.close(fig)
(OUT/'五折分析.json').write_text(json.dumps(dict(source=str(SRC),config=m['config'],summaries=summaries,rows=allrows),ensure_ascii=False,indent=2))
lines=['# 四组完整五折训练诊断','','实验已完成20个折模型，均200轮。只分析开发集交叉验证；最终模型和测试集尚未运行。','', '| 配置 | 五折宏F1均值±样本标准差 | 折外正确数 | 最佳轮次 | 150→200轮历史最佳F1平均增益 |','|---|---:|---:|---|---:|']
for s in summaries:lines.append(f'|{s["name"]}|{s["f1_mean"]:.2%} ± {s["f1_std"]:.2%}|{s["oof_correct"]}/{s["oof_total"]}|{s["best_epoch"]}|{s["f1gain150_200_mean"]*100:.2f}个百分点|')
lines+=['','## 早停检查','','实际早停监控验证损失，而最佳模型按验证宏F1选择（相同时按损失）。损失刷新历史最低至少0.0001时重置耐心计数；连续30轮不满足才早停。学习率调度监控验证损失，耐心8轮，衰减0.5。','']
for s in summaries:lines.append(f'- {s["name"]}：150轮平均验证损失{s["val_loss150_mean"]:.4f}，200轮{s["val_loss200_mean"]:.4f}；五折历史最长停滞{s["max_stale"]}轮；触发降学习率的折数{s["lr_reduced_folds"]}/5。')
lines+=['','## 后续建议','','先固定数据版本、划分和所有其他参数，原四组将上限提高到400轮（保留早停30轮），比较第200轮前后历史最佳验证F1和损失。优先验证128两组；若不补跑64组相同预算，不把原200轮排名当作充分收敛后的容量结论。确认训练预算后再单独比较mean/mean_std汇总或不同感受野。只做方案，不修改脚本或启动训练。','']
(OUT/'训练参数评估.md').write_text('\n'.join(lines))
