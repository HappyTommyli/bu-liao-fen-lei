"""Read-only descriptive signal analysis; no fitting or holdout evaluation."""
from pathlib import Path
import sys,json,csv,collections
from datetime import datetime
from zoneinfo import ZoneInfo
from matplotlib import font_manager
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.dataset import load_dataset
from cloth_trainer.data import prepare_trials,file_hash,load_record
from cloth_trainer.config import DataConfig
from cloth_trainer.evaluation import classification_metrics
DEST=Path(__file__).parent;TASK=ROOT/'outputs/models/CLOTH4_20261118_043041_a4aed4'
meta=json.loads((TASK/'metadata.json').read_text());records,snapshot=load_dataset(TASK/'dataset')
entries=snapshot['admission']['batches'][0]['entries'];entrymap={e['path']:e for e in entries}
assert len(entries)==300 and len(records)==291
rows=[];curves={};rawmeta={}
for e in entries:
    p=Path(e['path']);m=json.loads((p.parent/'metadata.json').read_text());rawmeta[str(p)]=m
    assert file_hash(p)==e['sha256'] and file_hash(p.parent/'metadata.json')==e['metadata_sha256']
    d=e['diagnostics']['direction_quality'];bas=m.get('baseline_B0_uT',{});cfg=m['context']['config']
    row=dict(path=str(p),record=p.stem,label=e['label'],sequence=m.get('sequence'),quality=e['audit']['status'],eligible=e['audit']['eligible'],history=e['audit'].get('history'),reasons=e['audit']['reasons'],
             created_at=m['created_at'],date=datetime.fromisoformat(m['created_at']).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat(),session=m['context'].get('capture_session'),batch=m['context'].get('batch',{}).get('id'),
             speed=cfg['x_speed'],x_ppmm=cfg['x_ppmm'],z_ppmm=cfg.get('z_ppmm'),z_min=cfg.get('z_min'),z_max=cfg.get('z_max'),baseline_By=bas['By_uT'],baseline_sigma_By=e['diagnostics']['baseline_sigma'][1],
             full_out=d['full_medians'][0],full_back=d['full_medians'][1],full_rate=d['full_rate'],common_out=d['common_medians'][0],common_back=d['common_medians'][1],common_rate=d['common_rate'],common_difference=d['common_difference'],
             same_sign_fraction=d['same_sign_fraction'],marked=d['marked'],bins=d['bins'],conditions=d['conditions'])
    pairs=np.array(d['bin_medians']);row['spatial_correlation']=float(np.corrcoef(pairs.T)[0,1]) if np.all(pairs.std(0)>1e-9) else None
    rows.append(row)
descriptive_records=records+[load_record(e['path'],DataConfig(**meta['config']['data']),signals=('by',)) for e in entries if not e['audit']['eligible']]
for r in descriptive_records:
    rr=next(x for x in rows if x['path']==r.path);tt=prepare_trials([r],'by');med=[np.median(t.values[:,0]) for t in tt]
    rr.update(amplitude=float(np.mean(med)),out_amplitude=float(med[0]),back_amplitude=float(med[1]),
        temporal_std=float(np.mean([np.std(t.values[:,0]) for t in tt])),variation=float(np.mean([np.mean(np.abs(np.diff(t.values[:,0]))) for t in tt])),
        lengths=[len(t.values) for t in tt],native_fs=[x['observed_fs'] for x in r.diagnostics['integrity']['stages'].values()] if 'stages' in r.diagnostics['integrity'] else None)
    curves[r.path]=[np.array([np.mean(part) for part in np.array_split(t.values[:,0],64)]) for t in tt]
labels=meta['labels'];summaries=[]
for label in labels:
    allrows=[r for r in rows if r['label']==label];rr=[r for r in allrows if r['eligible']];a=np.array([r['amplitude'] for r in rr]);early=[r['amplitude'] for r in rr if r['sequence']<=10];late=[r['amplitude'] for r in rr if r['sequence']>10]
    q=np.quantile(a,[.25,.5,.75]);earlymedian=float(np.median(early));latemedian=float(np.median(late));mean=float(a.mean());std=float(a.std(ddof=1))
    earlycv=float(np.std(early,ddof=1)/np.mean(early));latecv=float(np.std(late,ddof=1)/np.mean(late))
    s=dict(early_cv=earlycv,late_cv=latecv,label=label,total=len(allrows),eligible=len(rr),excluded=sum(not r['eligible'] for r in allrows),review=sum(r['quality']=='suggest_review' for r in allrows),
           median=float(q[1]),q25=float(q[0]),q75=float(q[2]),min=float(a.min()),max=float(a.max()),mean=mean,std=std,cv=std/mean,iqr_relative=float((q[2]-q[0])/q[1]),
           early_n=len(early),late_n=len(late),early_median=earlymedian,late_median=latemedian,order_shift=2*(latemedian-earlymedian)/(latemedian+earlymedian),late_early_ratio=latemedian/earlymedian,
           common_rate_median=float(np.median([r['common_rate'] for r in rr])),common_rate_max=float(max(r['common_rate'] for r in rr)),
           dates=sorted({r['date'] for r in allrows}),batch_count=len({r['batch'] for r in allrows}),baseline_noise_median=float(np.median([r['baseline_sigma_By'] for r in rr])))
    summaries.append(s)
valid=[r for r in rows if r['eligible']];pairs=[]
for i,a in enumerate(summaries):
 for b in summaries[i+1:]:
    overlap=max(0,min(a['q75'],b['q75'])-max(a['q25'],b['q25']))
    pairs.append(dict(a=a['label'],b=b['label'],median_gap=abs(a['median']-b['median']),standardized_mean_gap=abs(a['mean']-b['mean'])/np.sqrt((a['std']**2+b['std']**2)/2),iqr_overlap=overlap))
# Model metrics use validation predictions only; no held-out model evaluation.
cv={}
for arch in ('gru','tcn','cera'):
    models=[r for r in meta['models'].values() if r['architecture']==arch and r['signal']=='by' and r['readout']=='mean'];vr=[r for m in models for r in m['val_predictions']]
    assert len({r['id'] for r in vr})==len(vr)
    metrics=classification_metrics(vr,labels);matrix=np.array(metrics['confusion']);confusions=[]
    for i in range(len(labels)):
     for j in range(i+1,len(labels)):
      if matrix[i,j]+matrix[j,i]:confusions.append(dict(a=labels[i],b=labels[j],count=int(matrix[i,j]+matrix[j,i]),a_to_b=int(matrix[i,j]),b_to_a=int(matrix[j,i])))
    cv[arch]=dict(fold_mean=meta['summaries'][arch+'_by_mean'],pooled_validation=metrics,confusions=sorted(confusions,key=lambda x:-x['count']),directions={d:classification_metrics([r for r in vr if r['direction']==d],labels) for d in ('out','back')})
for s in summaries:
 for arch in cv:s[arch+'_val_f1']=next(p['f1'] for p in cv[arch]['pooled_validation']['per_class'] if p['label']==s['label'])
effect={}
for key in ('amplitude','temporal_std','variation'):
    allv=np.array([r[key] for r in valid]);grand=allv.mean();total=float(np.sum((allv-grand)**2));between=sum(len([r for r in valid if r['label']==c])*(np.mean([r[key] for r in valid if r['label']==c])-grand)**2 for c in labels)
    effect[key]=dict(between_class_fraction=float(between/total),within_class_fraction=float(1-between/total))
b0=np.array([r['baseline_By']-np.median([x['baseline_By'] for x in valid if x['label']==r['label']]) for r in valid]);amp=np.array([r['amplitude']-np.median([x['amplitude'] for x in valid if x['label']==r['label']]) for r in valid]);rho=float(spearmanr(b0,amp).statistic)
result=dict(scope=dict(task=str(TASK),csvs=300,eligible=291,excluded=9,labels=15,signal='|ΔBy|',contact_mm=dict(out=[45,142],back=[16,114]),raw_sources_verified=True,formal_test_evaluated=False,descriptive_scope_includes_reserved_inputs=True),
    quality=dict(collections.Counter(r['quality'] for r in rows)),classes=summaries,records=rows,
    direction=dict(eligible_median=float(np.median([r['common_rate'] for r in valid])),eligible_p90=float(np.quantile([r['common_rate'] for r in valid],.9)),eligible_max=float(max(r['common_rate'] for r in valid)),all_over10=sum(r['common_rate']>=.1 for r in rows),eligible_over10=sum(r['common_rate']>=.1 for r in valid),spatial_correlation_median=float(np.median([r['spatial_correlation'] for r in valid if r['spatial_correlation'] is not None]))),
    amplitude_pairs=sorted(pairs,key=lambda x:x['standardized_mean_gap']),effect=effect,baseline_centered_spearman=rho,
    metadata=dict(speeds=sorted({r['speed'] for r in rows}),x_ppmm=sorted({r['x_ppmm'] for r in rows}),z_limits=sorted({str((r['z_min'],r['z_max'])) for r in rows}),dates=sorted({r['date'] for r in rows}),lengths=sorted({n for r in valid for n in r['lengths']})),cv=cv)
(DEST/'分析结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with (DEST/'各布料差异统计.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['布料','导入CSV','可训练CSV','排除CSV','复核CSV','幅值中位数μT','幅值下四分位μT','幅值上四分位μT','幅值最小μT','幅值最大μT','同类变异系数','001至010幅值中位数μT','011至020幅值中位数μT','前后段相对差异','共同范围去回差异率中位数','CERA开发集F1','TCN开发集F1'])
    for s in summaries:w.writerow([s[k] for k in ('label','total','eligible','excluded','review','median','q25','q75','min','max','cv','early_median','late_median','order_shift','common_rate_median','cera_val_f1','tcn_val_f1')])
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fontname=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[fontname,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(15,7.5),gridspec_kw={'width_ratios':[1.35,1]})
positions=np.arange(len(labels));rng=np.random.default_rng(42)
for i,label in enumerate(labels):
    rr=[r for r in valid if r['label']==label];values=np.array([r['amplitude'] for r in rr]);axes[0].scatter(values,i+rng.uniform(-.12,.12,len(rr)),s=15,color='#3478ba',alpha=.7)
    s=summaries[i];axes[0].plot([s['q25'],s['q75']],[i,i],lw=4,color='#164d7a');axes[0].scatter(s['median'],i,s=45,color='#e29220',zorder=4)
    for r in rows:
      if r['label']==label and not r['eligible']:axes[0].scatter(r['amplitude'],i,marker='x',s=55,c='#c64545')
    axes[1].scatter([100*r['common_rate'] for r in rr],i+rng.uniform(-.12,.12,len(rr)),s=15,color='#3478ba',alpha=.7)
    bad=[r for r in rows if r['label']==label and not r['eligible']]
    axes[1].scatter([100*r['common_rate'] for r in bad],[i]*len(bad),marker='x',s=55,color='#c64545')
for ax in axes:
    ax.set_yticks(positions,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].set_title('同类波动与不同布料的幅值重叠');axes[0].set_xlabel('完整有效段 |ΔBy| 中位数的往返平均 / μT')
axes[1].set_title('同一次采集的去回程差异');axes[1].set_xlabel('共同X范围幅值差异率 / %');axes[1].axvline(10,color='#777777',ls='--',lw=1);axes[1].set_xlim(left=0)
fig.suptitle('15类 × 前20个CSV：蓝点为可训练记录，红叉为排除记录\n幅值图深蓝线为中间50%范围，橙点为类中位数；去回差异不能单凭10%一项判排除',fontsize=12)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(DEST/'幅值与去回差异.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(15,7.5),gridspec_kw={'width_ratios':[1.5,1]})
heat=np.full((len(labels),20),np.nan)
for r in rows:
    val=r.get('amplitude',np.mean([r['full_out'],r['full_back']]))
    heat[labels.index(r['label']),r['sequence']-1]=val/summaries[labels.index(r['label'])]['median']
im=axes[0].imshow(heat,aspect='auto',cmap='coolwarm',vmin=.6,vmax=1.4);axes[0].set_yticks(positions,labels);axes[0].set_xticks(np.arange(0,20,2),np.arange(1,21,2));axes[0].set_xlabel('采集编号');axes[0].axvline(9.5,color='black',ls='--',lw=.8);axes[0].set_title('随采集编号的幅值变化（各类中位数=1）')
for r in rows:
 if not r['eligible']:axes[0].text(r['sequence']-1,labels.index(r['label']),'×',ha='center',va='center',color='black',fontsize=11)
fig.colorbar(im,ax=axes[0],fraction=.035,pad=.025,label='本次幅值 / 该类中位数（色阶限0.6～1.4）')
x=np.arange(len(labels));axes[1].plot([s['cera_val_f1'] for s in summaries],positions,'o-',label='CERA',color='#3478ba');axes[1].plot([s['tcn_val_f1'] for s in summaries],positions,'s-',label='TCN',color='#d58b21');axes[1].set_yticks(positions,labels);axes[1].invert_yaxis();axes[1].set_xlim(0,1.03);axes[1].grid(axis='x',alpha=.2);axes[1].set_xlabel('开发集五折汇总逐类F1');axes[1].set_title('同一实验的开发集验证表现');axes[1].legend(loc='lower left')
fig.suptitle('编号趋势与分类表现：只读取开发集验证结果，未运行最终留出测试',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(DEST/'编号变化与验证表现.png',dpi=160);plt.close(fig)
print('QUALITY',result['quality']);print('DIRECTION',result['direction']);print('META',result['metadata']);print('EFFECT',effect);print('B0RHO',rho)
for s in summaries:print('CLASS',s['label'],'n',s['eligible'],'median',round(s['median'],2),'range',round(s['min'],2),round(s['max'],2),'CV',round(s['cv'],3),'early/late',round(s['early_median'],2),round(s['late_median'],2),'shift%',round(s['order_shift']*100,2),'CERA F1',round(s['cera_val_f1'],3))
for arch in cv:print('CV',arch,cv[arch]['fold_mean'],'pooled',cv[arch]['pooled_validation']['macro_f1'],'confusions',cv[arch]['confusions'][:5])
print('AMPLITUDE_CLOSEST',pairs and sorted(pairs,key=lambda x:x['standardized_mean_gap'])[:5])
