"""Compare user-specified physical bounds with the earlier provisional time crop."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT=Path(__file__).resolve().parent
MANIFEST=Path('/home/serein/classify-cloth/outputs/20261114_contact_crop_preview_v1/manifest.json')
pair=next(p for p in json.loads(MANIFEST.read_text())['pairs'] if p['run']=='空气层_001')
source=Path(pair['source'])
meta=json.loads((source.parent/'metadata.json').read_text())
ppmm=meta['context']['config']['x_ppmm']
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==pair['source_sha256']
with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False})
fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
summary=dict(source=str(source),source_sha256=digest,pulses_per_mm=ppmm,user_bounds_mm=[44.,137.],
    note='Positions are latest motor software pulse counts, not synchronized encoder readings. Compare with provisional candidate-entry+3.5s/candidate-exit-0.5s crops; no boundaries or source data changed.',segments=[])

for i,(segment,ax) in enumerate(zip(pair['segments'],axs)):
    selected=[r for r in rows if r['stage']==segment['stage']]
    x=np.array([float(r['motor_x_pulses'])/ppmm for r in selected])
    t=np.array([int(r['device_time_unwrapped_us']) for r in selected],dtype=float)/1e6;t-=t[0]
    seq=np.array([int(r['seq']) for r in selected])
    y=np.array([float(r['dBy_uT']) for r in selected])
    if segment['stage']=='step_004':y=np.abs(y)
    valid=np.array([r['sample_valid']=='1' for r in selected])
    auto=(seq>=segment['first_seq'])&(seq<=segment['last_seq'])
    user=(x>=44)&(x<=137)
    auto_bounds=[float(x[auto].min()),float(x[auto].max())]
    overlap=max(0.,min(auto_bounds[1],137)-max(auto_bounds[0],44))
    union=max(auto_bounds[1],137)-min(auto_bounds[0],44)
    item=dict(direction=segment['direction'],stage=segment['stage'],
        old_auto_position_bounds_mm=auto_bounds,old_auto_time_bounds_s=[float(t[auto][0]),float(t[auto][-1])],
        user_position_bounds_mm=[44.,137.],
        observed_positions_in_user_selection_mm=[float(x[user].min()),float(x[user].max())],
        user_time_bounds_s=[float(t[user][0]),float(t[user][-1])],
        user_selected_rows=int(user.sum()),auto_selected_rows=int(auto.sum()),
        interval_intersection_over_union=overlap/union,
        auto_low_minus_user_low_mm=auto_bounds[0]-44,
        auto_high_minus_user_high_mm=auto_bounds[1]-137)
    summary['segments'].append(item)
    ax.axvspan(44,137,color='#279a60',alpha=.13,label='你给出的范围：44–137 mm')
    ax.axvspan(*auto_bounds,facecolor='#e99b32',edgecolor='#c78020',alpha=.16,hatch='///',label=f'之前自动保留：{auto_bounds[0]:.1f}–{auto_bounds[1]:.1f} mm')
    for bound in [44,137]:ax.axvline(bound,color='#248450',ls='--',lw=1.3)
    for bound in auto_bounds:ax.axvline(bound,color='#b97920',ls=':',lw=1.2)
    raw=y.copy();raw[~valid]=np.nan
    ax.plot(x,raw,color='#1768ac',lw=.45,alpha=.24,rasterized=True)
    centers=np.arange(.5,155,1.)
    median=[]
    for center in centers:
        mask=valid&(x>=center-.5)&(x<center+.5)
        median.append(float(np.median(y[mask])) if mask.any() else np.nan)
    ax.plot(centers,median,color='#1768ac',lw=1.5,label='磁信号（每 1 mm 中位数；浅线为原始值）')
    description='去程 step_003（向右运动 →）' if i==0 else '回程 step_004（← 向左运动）'
    ax.set(title=description,ylabel='ΔBy（µT）' if i==0 else '|ΔBy|（µT）',xlim=(0,155),ylim=(-30,520))
    ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower center',fontsize=9)
axs[-1].set_xlabel('相对零点的软件 X 位置（mm）；两幅图使用相同空间坐标')
axs[-1].set_xticks([0,20,44,60,80,100,120,137,155])
fig.suptitle('空气层_001：你的 44–137 mm 范围与之前自动裁剪范围对比\n绿色为你给出的范围，橙色斜线为之前的保守时间裁剪结果',fontsize=14)
fig.savefig(OUT/'position_bounds_comparison.png',dpi=170)
plt.close(fig)
(OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
print(json.dumps(summary,ensure_ascii=False,indent=2))
