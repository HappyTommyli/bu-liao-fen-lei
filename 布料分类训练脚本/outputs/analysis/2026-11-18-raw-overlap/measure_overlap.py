"""Quantify class overlap with fixed descriptive definitions, no fitted classifier."""
import json,csv,itertools
from pathlib import Path
import numpy as np
DEST=Path(__file__).parent
d=json.loads((DEST/'原始数据清单.json').read_text());cache=np.load(DEST/'原始波形统计缓存.npz')
p=cache['profiles'];f=cache['features'];valid=sorted([r for r in d['records'] if r['eligible']],key=lambda r:r['feature_index'])
labels=d['scope']['labels'];ind={label:np.array([i for i,r in enumerate(valid) if r['label']==label]) for label in labels}
q=lambda a:np.nanquantile(a,[.05,.5,.95],axis=0)
def band_overlap(qa,qb):
    intersection=np.maximum(0,np.minimum(qa[2],qb[2])-np.maximum(qa[0],qb[0]))
    union=np.maximum(qa[2],qb[2])-np.minimum(qa[0],qb[0])
    return float(np.sum(intersection)/np.sum(union)) if np.sum(union)>1e-12 else float(np.allclose(qa[1],qb[1]))
def interval_overlaps(qa,qb):
    ii=np.maximum(0,np.minimum(qa[2],qb[2])-np.maximum(qa[0],qb[0]));uu=np.maximum(qa[2],qb[2])-np.minimum(qa[0],qb[0])
    return np.divide(ii,uu,out=np.zeros_like(ii),where=uu>1e-12)
def measurement(aa,bb):
    qa,qb=q(aa),q(bb)
    distance=float(np.sqrt(np.mean((qa[1]-qb[1])**2)))
    ra=float(np.median(np.sqrt(np.nanmean((aa-qa[1])**2,axis=1))));rb=float(np.median(np.sqrt(np.nanmean((bb-qb[1])**2,axis=1))))
    return dict(overlap=band_overlap(qa,qb),distance=distance,scatter_a=ra,scatter_b=rb,separation=distance/(ra+rb) if ra+rb>1e-12 else None)
centers={};summary=[]
for label in labels:
    ix=ind[label];aa=p[ix];qq=q(aa);centers[label]=qq;ff=f[ix];fq=q(ff)
    repeat_rms=np.sqrt(np.nanmean((aa-qq[1])**2,axis=1));meanamp=float(np.mean(ff[:,0]))
    allrows=[r for r in d['records'] if r['label']==label]
    cohorts=[]
    for low,high in [(1,10),(11,20),(21,30),(31,35),(36,45)]:
        chosen=[i for i in ix if low<=valid[i]['sequence']<=high]
        if chosen:cohorts.append(dict(low=low,high=high,n=len(chosen),amplitude=float(np.median(f[chosen,0])),amplitude_cv=float(np.std(f[chosen,0],ddof=1)/np.mean(f[chosen,0])) if len(chosen)>1 else None))
    summary.append(dict(label=label,total=len(allrows),eligible=len(ix),unusable=len(allrows)-len(ix),amplitude=float(fq[1,0]),amplitude_low=float(fq[0,0]),amplitude_high=float(fq[2,0]),
        cv=float(np.std(ff[:,0],ddof=1)/meanamp),within_rms=float(np.median(repeat_rms)),radius90=float(np.quantile(repeat_rms,.9)),
        feature_q=fq.tolist(),cohorts=cohorts))
pairs=[]
for a,b in itertools.combinations(labels,2):
    ia,ib=ind[a],ind[b];aadata=p[ia];bbdata=p[ib];m=measurement(aadata,bbdata)
    fi=interval_overlaps(q(f[ia]),q(f[ib]));m.update(a=a,b=b,amplitude_interval_overlap=float(fi[0]),texture_overlap=fi[1:].tolist(),
        out_overlap=band_overlap(q(aadata[:,:48]),q(bbdata[:,:48])),back_overlap=band_overlap(q(aadata[:,48:]),q(bbdata[:,48:])))
    # Compare same software-X positions in both directions as a sensitivity check.
    common=np.r_[np.arange(34),48+np.arange(14,48)]
    m['common']=measurement(aadata[:,common],bbdata[:,common])
    shape=lambda v:np.concatenate((v[:,:48]-np.nanmedian(v[:,:48],axis=1,keepdims=True),v[:,48:]-np.nanmedian(v[:,48:],axis=1,keepdims=True)),axis=1)
    m['centered_shape']=measurement(shape(aadata),shape(bbdata))
    cohorts=[]
    for low,high in [(1,10),(11,20),(21,30),(31,35)]:
        ja=[i for i in ia if low<=valid[i]['sequence']<=high];jb=[i for i in ib if low<=valid[i]['sequence']<=high]
        if min(len(ja),len(jb))>=5:
            cm=measurement(p[ja],p[jb]);cm.update(low=low,high=high,n_a=len(ja),n_b=len(jb));cohorts.append(cm)
    m['cohorts']=cohorts
    # Fixed exploratory thresholds; these are not confidence/accuracy estimates.
    m['high_wave_overlap']=m['overlap']>=.50 and m['separation']<1.5
    m['texture_also_overlaps']=bool(np.min(fi[1:])>=.30)
    m['same_cohort_overlap_count']=sum(c['overlap']>=.50 and c['separation']<1.5 for c in cohorts)
    m['hard_pair']=bool(m['high_wave_overlap'] and m['texture_also_overlaps'])
    pairs.append(m)
for s in summary:
    related=sorted([r for r in pairs if s['label'] in (r['a'],r['b'])],key=lambda r:-r['overlap'])
    r=related[0];s.update(closest=r['b'] if r['a']==s['label'] else r['a'],max_overlap=r['overlap'],closest_distance=r['distance'],closest_separation=r['separation'],
        high_wave_partners=[r['b'] if r['a']==s['label'] else r['a'] for r in related if r['high_wave_overlap']],
        hard_partners=[r['b'] if r['a']==s['label'] else r['a'] for r in related if r['hard_pair']])
out=dict(scope=d['scope'],quality=d['quality'],parameters=dict(band_quantiles=[.05,.95],high_band_overlap=.5,high_scatter_ratio_below=1.5,texture_min_interval_iou=.3,
    warning='Exploratory descriptive thresholds, not probability, accuracy, or literature cutoffs.'),classes=summary,pairs=sorted(pairs,key=lambda r:-r['overlap']))
(DEST/'重合度分析.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('CLASSES')
for s in summary:print(s['label'],'n',s['eligible'],'amp',round(s['amplitude'],1),'cv',round(100*s['cv'],2),'rms',round(s['within_rms'],2),'closest',s['closest'],'ov',round(s['max_overlap'],3),'ratio',round(s['closest_separation'],2),'hard',s['hard_partners'])
print('TOP PAIRS')
for r in out['pairs'][:30]:print(r['a'],r['b'],'overlap',round(r['overlap'],3),'S',round(r['separation'],2),'texture',[round(x,2)for x in r['texture_overlap']],'cohort',[(x['low'],round(x['overlap'],2),round(x['separation'],2))for x in r['cohorts']],'hard',r['hard_pair'])
