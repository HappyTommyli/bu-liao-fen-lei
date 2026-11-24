"""Raw data only; reuse quality checks solely for identical CSV+metadata hashes."""
import sys,json,csv,collections
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from cloth_trainer.data import discover_report,load_record,file_hash
from cloth_trainer.config import DataConfig
from cloth_trainer.quality import RULE_VERSION
DEST=Path(__file__).parent;SOURCE=ROOT.parent/'xz-control/data/布料采集数据'
prior=DEST.parent/'2026-11-18-current-evaluation/逐次采集审查明细.csv'
old={r['CSV路径']:r for r in csv.DictReader(prior.open(encoding='utf-8-sig'))}
statusmap={'审查通过':'normal','可用·建议复核':'suggest_review','本实验排除':'excluded','无法判断':'undetermined','数据错误':'data_error'}
paths,ignored,errors=discover_report([SOURCE]);fabric=[];other=[]
for p in paths:
    m=json.loads((p.parent/'metadata.json').read_text())
    (other if m.get('label') in ('三角形间隔','矩形间隔') else fabric).append(p)
features=[];profiles=[];rows=[];fresh=0;reused=0
for j,p in enumerate(fabric):
    mp=p.parent/'metadata.json';meta=json.loads(mp.read_text());sha,md=file_hash(p),file_hash(mp)
    row=dict(path=str(p),label=meta.get('label'),record=p.stem,sequence=meta.get('sequence'),created_at=meta.get('created_at'),
       sha256=sha,metadata_sha256=md,status='pending',eligible=False,reasons=[])
    previous=old.get(str(p))
    if meta.get('outcome','completed')!='completed':row['reasons']=['采集未完整完成，本次暂不评估：'+str(meta.get('outcome'))]
    elif previous and previous['CSV指纹']==sha and previous['metadata指纹']==md:
        row.update(status=statusmap[previous['审查状态']],eligible=previous['可训练']=='是',reasons=[previous['处理原因']] if previous['处理原因'] else [])
        reused+=1
    else:
        fresh+=1
        try:
            record=load_record(p,DataConfig(),signals=('by',));row.update(status=record.quality,eligible=record.audit['eligible'],reasons=record.audit['reasons'])
        except (ValueError,OSError,KeyError,TypeError) as exc:row.update(status='data_error',reasons=[str(exc)])
    rows.append(row)
    if row['eligible']:
        with p.open(encoding='utf-8-sig',newline='') as stream:header=next(csv.reader(stream))
        cols=[header.index(k) for k in ('sensor_t_s','stage','motor_x_pulses','dBy_uT','sample_valid')]
        raw=np.loadtxt(p,delimiter=',',skiprows=1,usecols=cols,dtype='U32',encoding='utf-8-sig')
        t=raw[:,0].astype(float);stages=raw[:,1];x=raw[:,2].astype(float)/float(meta['context']['config']['x_ppmm']);y=np.abs(raw[:,3].astype(float));good=(raw[:,4]=='1')&np.isfinite(y)
        pp=[];ff=[]
        for stage,lo,hi,rawlo,rawhi in [('step_003',45,141,45,142),('step_004',17,113,16,114)]:
            use=good&(stages==stage)&(x>=lo)&(x<hi)
            for left in range(lo,hi,2):
                values=y[use&(x>=left)&(x<left+2)];pp.append(float(np.median(values)) if len(values)>=3 else float('nan'))
            use=good&(stages==stage)&(x>=rawlo)&(x<=rawhi);tt=t[use];vraw=y[use]
            n=int(np.floor(((rawhi-rawlo)/6.-.2)*333));grid=tt[0]+np.arange(n)/333.
            pos=np.searchsorted(tt,grid,side='right')-1;assert pos.min()>=0 and grid[-1]<=tt[-1]+1e-9
            v=vraw[pos];residual=(v-uniform_filter1d(v,size=167,mode='nearest'))[83:-83]
            freq,power=welch(residual,fs=333.,nperseg=1024)
            integral=lambda a,b:float(np.trapezoid(power[(freq>=a)&(freq<b)],freq[(freq>=a)&(freq<b)]))
            ff.append([float(np.median(vraw)),float(np.sqrt(np.mean(residual**2))),float(np.mean(np.abs(np.diff(v)))),integral(1,20),integral(20,100)])
        feat=np.mean(ff,axis=0)
        if file_hash(p)!=sha or file_hash(mp)!=md:
            row.update(status='pending',eligible=False,reasons=['文件在读取中变化，本次暂不评估'])
        else:
            row.update(feature_index=len(profiles),empty_position_bins=int(np.isnan(pp).sum()),amplitude=float(feat[0]),residual_rms=float(feat[1]),mean_abs_diff=float(feat[2]),power_1_20=float(feat[3]),power_20_100=float(feat[4]))
            profiles.append(pp);features.append(feat)
    if (j+1)%50==0 or j+1==len(fabric):print('RAW',j+1,len(fabric),'fresh',fresh,'reused',reused,flush=True)
for row in rows:
    if file_hash(row['path'])!=row['sha256'] or file_hash(Path(row['path']).parent/'metadata.json')!=row['metadata_sha256']:
        row.update(status='pending',eligible=False,reasons=['读取后来源已变化，本次暂不评估'])
keep=[r['feature_index'] for r in rows if r['eligible']]
profiles=np.array(profiles)[keep];features=np.array(features)[keep]
for i,r in enumerate([r for r in rows if r['eligible']]):r['feature_index']=i
np.savez_compressed(DEST/'原始波形统计缓存.npz',profiles=profiles,features=features)
d=dict(scope=dict(source=str(SOURCE),csv_count=len(fabric),other_count=len(other),ignored_count=len(ignored),eligible=len(keep),
    labels=sorted({r['label'] for r in rows}),rule_version=RULE_VERSION,signal='|ΔBy|',bounds=dict(out=[45,141],back=[17,113]),bin_mm=2,
    feature_names=['amplitude','residual_rms','mean_abs_diff','power_1_20','power_20_100'],model_results_used=False,
    source_fingerprints_verified=True,quality_cache_exact_matches=reused,quality_fresh_reviews=fresh),
    quality=dict(collections.Counter(r['status'] for r in rows)),records=rows)
(DEST/'原始数据清单.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print('SCOPE',d['scope']);print('QUALITY',d['quality'])
