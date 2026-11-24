"""Compare absolute XY magnetic signals with time normalization only."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/serein/classify-cloth/xz-control/data/布料采集数据/空气层_001/空气层_001.csv')
original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'axes.unicode_minus':False})
NAMES = ['|ΔBx|', '|ΔBy|', 'ΔBxy = √(ΔBx² + ΔBy²)']
COLORS = ['#1768ac', '#d46b24']

def load_leg(stage, position_range):
    selected = [r for r in rows if r['stage'] == stage and r['sample_valid'] == '1']
    if position_range is not None:
        lo, hi = position_range
        selected = [r for r in selected if lo <= float(r['motor_x_pulses']) / 250 <= hi]
    t = np.array([int(r['device_time_unwrapped_us']) for r in selected], dtype=float) / 1e6
    assert np.all(np.diff(t) > 0)
    tau = (t-t[0])/(t[-1]-t[0])
    xy = np.array([[float(r['dBx_uT']), float(r['dBy_uT'])] for r in selected])
    values = np.column_stack([np.abs(xy), np.linalg.norm(xy, axis=1)])
    # Fixed equal-time bin means show the trend without amplitudes being rescaled.
    ids = np.minimum((tau*128).astype(int), 127)
    count = np.bincount(ids, minlength=128)
    assert np.all(count > 0)
    bins = np.column_stack([np.bincount(ids, weights=values[:,i], minlength=128)/count for i in range(3)])
    return dict(tau=tau, values=values, bins=bins,
        info=dict(stage=stage,n=len(selected), first_seq=int(selected[0]['seq']), last_seq=int(selected[-1]['seq']),
            first_sensor_t_s=float(selected[0]['sensor_t_s']),last_sensor_t_s=float(selected[-1]['sensor_t_s']),
            duration_s=float(t[-1]-t[0]),mean=values.mean(axis=0).tolist(),std=values.std(axis=0).tolist()))

results = dict(source=str(SOURCE), source_sha256=original_hash,
    transforms='Absolute dBx and dBy, and hypot(dBx,dBy). Normalize each selected leg to tau=(t-t0)/(t1-t0). No amplitude normalization, no new baseline subtraction, no fitted correction.',
    selection_note='40–110 mm is a common interior region visibly within the response plateau in the preceding plot. It is a comparison window, not an annotated full contact interval. Positions are latest software motor counts, not hardware-synchronized.',
    display='Thin lines: raw absolute values. Thick lines: 128 equal-time-bin means. Right column additionally reverses return time; left column is actual acquisition order.', regions={})

for key, bounds, title, filename in [
    ('common_interior', (40,110), '相同扫描中段 X=40–110 mm', 'abs_time_contact.png'),
    ('full_leg', None, '完整单程（包含悬空与接触过渡，仅作参照）', 'abs_time_full.png')]:
    legs = [load_leg(stage,bounds) for stage in ['step_003','step_004']]
    f, r = legs
    metrics = dict(outward=f['info'],return_leg=r['info'],channels={})
    for k,name in enumerate(NAMES):
        fm,rm = float(f['values'][:,k].mean()),float(r['values'][:,k].mean())
        metrics['channels'][name] = dict(outward_mean_uT=fm, return_mean_uT=rm,
            relative_mean_difference_percent=100*(rm-fm)/fm,
            mean_absolute_difference_128bins_acquisition_order_uT=float(np.abs(f['bins'][:,k]-r['bins'][:,k]).mean()),
            mean_absolute_difference_128bins_return_reversed_uT=float(np.abs(f['bins'][:,k]-r['bins'][::-1,k]).mean()))
    results['regions'][key] = metrics
    fig,axs=plt.subplots(3,2,figsize=(12.8,8.3),constrained_layout=True)
    grid=(np.arange(128)+.5)/128
    for k,name in enumerate(NAMES):
        for col in [0,1]:
            for j,(leg,color,label) in enumerate(zip(legs,COLORS,['去程','回程'])):
                reversed_return=col==1 and j==1
                x=1-leg['tau'] if reversed_return else leg['tau']
                y=leg['values'][:,k]
                trend=leg['bins'][::-1,k] if reversed_return else leg['bins'][:,k]
                axs[k,col].plot(x,y,color=color,alpha=.18,lw=.45,rasterized=True)
                axs[k,col].plot(grid,trend,color=color,lw=1.55,label=label+('（时间反向）' if reversed_return else ''))
            axs[k,col].set(xlim=(0,1),ylabel=f'{name}（µT）')
            axs[k,col].grid(alpha=.2)
            axs[k,col].spines[['top','right']].set_visible(False)
            axs[k,col].legend(loc='best',fontsize=9)
        lo,hi=min(axs[k,0].get_ylim()[0],axs[k,1].get_ylim()[0]),max(axs[k,0].get_ylim()[1],axs[k,1].get_ylim()[1])
        for col in [0,1]:axs[k,col].set_ylim(lo,hi)
    axs[0,0].set_title('取绝对值＋时间归一化，保持采集顺序',fontsize=12)
    axs[0,1].set_title('额外将回程时间反向，比较相同空间顺序',fontsize=12)
    axs[2,0].set_xlabel('归一化时间 τ（0=该段开始，1=该段结束）')
    axs[2,1].set_xlabel('归一化进度（回程另作时间反向）')
    fig.suptitle('空气层_001：'+title+'\n细线为原始数据，粗线为128个等时间箱均值；纵轴未归一化',fontsize=14)
    fig.savefig(OUT/filename,dpi=160)
    plt.close(fig)

(OUT/'abs_time_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash
print(json.dumps(results,ensure_ascii=False,indent=2))
