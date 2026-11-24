"""Read-only boundary feasibility audit. Candidate intervals are not ground truth."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import median_filter, binary_closing

ROOT=Path('/home/serein/classify-cloth/xz-control/data/布料采集数据')
OUT=Path(__file__).resolve().parent
P=65536*.242
FS=1000/3
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False})

def longest_interval(time, signal, threshold):
    mask=signal>threshold
    mask=binary_closing(mask, structure=np.ones(67,dtype=bool))
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
    intervals=[(a,b) for a,b in zip(edges[::2],edges[1::2]) if time[b-1]-time[a]>=5]
    if not intervals:return None
    a,b=max(intervals,key=lambda pair:time[pair[1]-1]-time[pair[0]])
    return dict(start_s=float(time[a]),end_s=float(time[b-1]),duration_s=float(time[b-1]-time[a]))

def noise_threshold(signal, time, floor):
    air=(time<1)|(time>time[-1]-1)
    base=float(np.median(signal[air]))
    sigma=float(1.4826*np.median(abs(signal[air]-base)))
    return base+max(6*sigma,floor)

records=[]
examples={}
sources=[]
for path in sorted(ROOT.glob('*/metadata.json')):
    meta=json.loads(path.read_text())
    source=path.parent/meta['files']['sensor']
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    sources.append(dict(path=str(source),sha256=digest))
    with source.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    for stage,direction in [('step_003','去程'),('step_004','回程')]:
        subset=[r for r in rows if r['stage']==stage]
        t=np.array([int(r['device_time_unwrapped_us']) for r in subset],dtype=float)/1e6
        t-=t[0]
        assert np.all(np.diff(t)>0)
        xyz=np.array([[float(r[f'B{a}_uT']) for a in 'xyz'] for r in subset])
        delta=np.array([[float(r[f'dB{a}_uT']) for a in 'xyz'] for r in subset])
        valid=np.array([r['sample_valid']=='1' for r in subset])
        x=np.array([float(r['motor_x_pulses'])/250 for r in subset])
        center=valid & (x>=50)&(x<=100)
        first=valid & (t<1)
        last=valid & (t>t[-1]-1)
        z=xyz[:,2]
        # Diagnostic only, conditional on a signed-16 period at documented settings.
        z_unwrapped=np.unwrap(z*2*np.pi/P)*P/(2*np.pi)
        grid=np.arange(0,t[-1],1/FS)
        z_regular=np.interp(grid,t[valid],z_unwrapped[valid])
        # User describes both ends of each single leg as airborne. This remains an assumption.
        z_air_start=float(np.median(z_regular[grid<1]))
        z_air_end=float(np.median(z_regular[grid>grid[-1]-1]))
        baseline=np.interp(grid,[.5,grid[-1]-.5],[z_air_start,z_air_end])
        z_score=median_filter(abs(z_regular-baseline),size=51,mode='nearest')
        xy=np.hypot(delta[:,0],delta[:,1])
        xy_regular=np.interp(grid,t[valid],xy[valid])
        xy_score=median_filter(xy_regular,size=51,mode='nearest')
        z_threshold=noise_threshold(z_score,grid,30.)
        xy_threshold=noise_threshold(xy_score,grid,50.)
        z_candidate=longest_interval(grid,z_score,z_threshold)
        xy_candidate=longest_interval(grid,xy_score,xy_threshold)
        raw_score=median_filter(np.interp(grid,t[valid],abs(delta[valid,2])),size=51,mode='nearest')
        raw_candidate=longest_interval(grid,raw_score,1000.)
        entry=dict(run=meta['run_name'],label=meta['label'],direction=direction,stage=stage,
            source=str(source),rows=len(subset),invalid_samples=int((~valid).sum()),
            raw_Bz_median_air_start_uT=float(np.median(z[first])),raw_Bz_median_middle_uT=float(np.median(z[center])),
            raw_Bz_median_air_end_uT=float(np.median(z[last])),
            dBz_median_middle_uT=float(np.median(delta[center,2])),
            dBy_median_middle_uT=float(np.median(delta[center,1])),
            jumps_gt10000uT=int((abs(np.diff(z))>10000).sum()),
            raw_abs_dBz_gt1000_candidate=raw_candidate,
            diagnostic_z_threshold_uT=z_threshold,diagnostic_z_candidate=z_candidate,
            xy_threshold_uT=xy_threshold,xy_candidate=xy_candidate)
        records.append(entry)
        if meta['sequence']==1:
            examples[(meta['label'],direction)]=dict(time=t,raw_z=z,xy=xy,entry=entry)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest

summary=[]
for label in sorted({r['label'] for r in records}):
    for direction in ['去程','回程']:
        rs=[r for r in records if r['label']==label and r['direction']==direction]
        def bounds(field):return [round(min(r[field] for r in rs),3),round(max(r[field] for r in rs),3)]
        summary.append(dict(label=label,direction=direction,count=len(rs),
            middle_dBz_range_uT=bounds('dBz_median_middle_uT'),
            raw_large_threshold_candidates=sum(r['raw_abs_dBz_gt1000_candidate'] is not None for r in rs),
            diagnostic_z_candidates=sum(r['diagnostic_z_candidate'] is not None for r in rs),
            xy_candidates=sum(r['xy_candidate'] is not None for r in rs)))

payload=dict(method='Exploratory boundary proposal, no classification or segmentation accuracy claimed.',
    assumptions=['First and last 1 s of each leg are airborne, per user description.',
        'Bz unwrapping is diagnostic only, conditioned on 65536*0.242 uT period; original data are unchanged.',
        'All threshold rules are shared across labels; no material name used in candidate detection.'],
    candidate_rule='Median 51 samples (~0.15 s), close gaps <=~0.2 s, retain longest interval >=5 s. XY threshold=air median+max(6*MAD_sigma,50uT). Diagnostic unwrapped Z first removes a linear endpoint baseline; threshold=air median+max(6*MAD_sigma,30uT).',
    limitation='No hardware contact trigger or manual ground-truth intervals. A candidate can contain contact transition, and is not yet a steady-training segment.',
    summary=summary,records=records,sources=sources)
(OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

labels=['空气层','织棉提花缎','岩石肌理面料']
fig,axs=plt.subplots(3,2,figsize=(13,8),constrained_layout=True)
for i,label in enumerate(labels):
    for j,direction in enumerate(['去程','回程']):
        ex=examples[(label,direction)]
        ax=axs[i,j]
        ax.plot(ex['time'],ex['raw_z'],lw=.65,color='#7155a5',rasterized=True)
        if direction=='回程':
            lo,hi=np.quantile(ex['raw_z'],[.005,.995])
            margin=max(20.,.08*(hi-lo))
            ax.set_ylim(lo-margin,hi+margin)
        candidate=ex['entry']['xy_candidate']
        if candidate:ax.axvspan(candidate['start_s'],candidate['end_s'],color='#3b9a66',alpha=.13,label='XY 响应候选接触区间')
        ax.set(title=f'{label}_001 · {direction}',xlabel='该单程开始后时间（s）',ylabel='原始 Bz（µT）')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='best',fontsize=8)
fig.suptitle('Bz 在去程与回程中的表现不同（各子图独立纵轴）\n回程放大显示主体，越界尖峰省略；浅绿为 XY 接触候选，未经人工标注',fontsize=14)
fig.savefig(OUT/'bz_boundary_comparison.png',dpi=160)
plt.close(fig)

fig,axs=plt.subplots(3,2,figsize=(13,7.6),constrained_layout=True)
for i,label in enumerate(labels):
    for j,direction in enumerate(['去程','回程']):
        ex=examples[(label,direction)];ax=axs[i,j]
        ax.plot(ex['time'],ex['xy'],lw=.6,color='#1768ac',rasterized=True)
        ax.axhline(ex['entry']['xy_threshold_uT'],ls='--',lw=.9,color='#bd6931',label='当前候选阈值')
        candidate=ex['entry']['xy_candidate']
        if candidate:ax.axvspan(candidate['start_s'],candidate['end_s'],color='#3b9a66',alpha=.13,label='候选接触区间')
        ax.set(title=f'{label}_001 · {direction}',xlabel='该单程开始后时间（s）',ylabel='XY 模长（µT）')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(loc='best',fontsize=8)
fig.suptitle('XY 模长提供两个方向共有的候选接触线索\n候选区间仍包含上下平台过渡，正式训练前需要进一步裁剪',fontsize=14)
fig.savefig(OUT/'xy_contact_candidates.png',dpi=160)
plt.close(fig)

lines=['# 现有记录的接触边界可辨性检查','',
    '检查 25 份完整往返、50 个单程。只读原始记录；不将候选检出率当作分段准确率。','',
    '|布料|方向|数量|平台中段 ΔBz 中位数范围 µT|大阈值候选数|诊断 Z 候选数|XY 候选数|',
    '|---|---|---:|---|---:|---:|---:|']
for r in summary:
    lines.append(f"|{r['label']}|{r['direction']}|{r['count']}|{r['middle_dBz_range_uT']}|{r['raw_large_threshold_candidates']}|{r['diagnostic_z_candidates']}|{r['xy_candidates']}|")
lines += ['', '大阈值示例：|ΔBz| > 1000 µT，候选至少持续 5 s。它是检验“大幅 Bz 跳变”假设的例子，不是冻结阈值。',
    '', '平台中段数值来自软件 X=50–100 mm；此前曲线显示其处于各记录的响应平台内。没有独立接触标签。',
    '', '## 结论', '',
    '- Bz 的变化可提供接触线索，但去程的大幅跳变不能直接作为两个方向通用的阈值。',
    '- 所有去程的大跳变与先前疑似 signed16 量程回绕一致，不能直接视为真实的大磁场变化。',
    '- 回程 Bz 中段变化约几十至一百多 µT；与去程大小和方向均不同。',
    '- XY 共有的持续响应可用来提出接触候选区间，Bz 可作辅助验证。',
    '- 所有候选都需要检查进入后的稳定时间与离开前边缘，不能把刚接触到离开之间全段直接当稳定训练段。',
    '- 若使用 Bz 处理历史记录，解回绕只能作为有条件诊断；应先确认传感器配置与实际输出机制。',
    '- 原始文件 SHA256 记录在 results.json，并在读取前后核对。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('TOTAL',len(records),'raw_large',sum(r['raw_abs_dBz_gt1000_candidate'] is not None for r in records),'diagnostic_z',sum(r['diagnostic_z_candidate'] is not None for r in records),'xy',sum(r['xy_candidate'] is not None for r in records))
