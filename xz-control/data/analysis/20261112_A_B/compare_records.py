"""Read-only A/B comparison; outputs summaries and plots beside this script."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path('/home/serein/classify-cloth/xz-control/data/experiments_linux')
OUT=Path(__file__).resolve().parent
RUNS={'A':'20261112_185152_549605','B':'20261112_185257_470986'}
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
font_name=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'A':'#227c9d','B':'#d56b32'}
data={}; result={}
for label,run in RUNS.items():
    path=ROOT/run/'sensor.csv'
    with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    metadata=json.loads(path.with_name('metadata.json').read_text())
    t=np.array([float(r['sensor_t_s']) for r in rows])
    raw=np.array([[float(r[c]) for c in ('Bx_uT','By_uT','Bz_uT')] for r in rows])
    valid=np.array([r['sample_valid']=='1' for r in rows]); stage=np.array([r['stage'] for r in rows])
    forward=stage=='step_001'; start=t[forward][0]; end=t[forward][-1]
    # Same direction and constant-speed interior; trim 1 s from each end.
    central=forward&valid&(t>=start+1)&(t<=end-1)
    tc=t[central]-start; bc=raw[central]
    # Interpolate only analysis copies to a uniform time grid for the filter.
    # No original CSV values or flags are changed.
    fs=(len(t)-1)/(t[-1]-t[0]); grid=np.arange(tc[0],tc[-1],1/fs)
    uniform=np.column_stack([np.interp(grid,tc,bc[:,i]) for i in range(3)])
    window=int(round(.5*fs))|1
    residual=uniform-signal.savgol_filter(uniform,window,2,axis=0)
    jumps=np.flatnonzero(np.abs(np.diff(raw[:,2]))>10000)
    record=dict(run=run,csv=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=len(rows),
        duration_s=float(t[-1]-t[0]),sampling_rate_hz=float(fs),invalid_rows=int((~valid).sum()),
        clock_segments=sorted(set(int(r['device_clock_segment']) for r in rows)),
        dt_median_ms=float(np.median(np.diff(t))*1000),dt_max_ms=float(np.max(np.diff(t))*1000),
        gaps_gt_4p5ms=int((np.diff(t)>.0045).sum()),
        stage_rows={s:int((stage==s).sum()) for s in dict.fromkeys(stage)},
        central_n=int(central.sum()),central_interval_from_forward_start_s=[float(tc[0]),float(tc[-1])],
        central_mean_uT=bc.mean(0).tolist(),central_std_uT=bc.std(0).tolist(),
        central_p95_minus_p5_uT=(np.percentile(bc,95,axis=0)-np.percentile(bc,5,axis=0)).tolist(),
        residual_rms_uT=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        baseline_sigma_uT=metadata['context']['baseline']['sigma'],
        bz_large_jumps=int(len(jumps)),bz_large_jump_sizes_uT=np.diff(raw[:,2])[jumps].tolist(),
        bz_large_jump_times_s=t[jumps].tolist(),metadata=metadata)
    # Six disjoint one-second blocks are descriptive, not independent fabric trials.
    windows=[]
    for a in range(1,7):
        mask=(grid>=a)&(grid<a+1)
        if mask.sum()>10:
            windows.append(dict(start_s=a,residual_rms_uT=np.sqrt(np.mean(residual[mask]**2,axis=0)).tolist()))
    record['within_record_1s_windows']=windows
    result[label]=record
    data[label]=dict(t=t,raw=raw,forward=forward,start=start,grid=grid,residual=residual)

fig,axes=plt.subplots(2,2,figsize=(13,7.8),layout='constrained')
for i,axis in enumerate(('Bx','By')):
    for label,d in data.items():
        axes[i,0].plot(d['t'][d['forward']]-d['start'],d['raw'][d['forward'],i],color=COLORS[label],lw=.8,label=label,alpha=.9)
        axes[i,1].plot(d['grid'],d['residual'][:,i],color=COLORS[label],lw=.7,label=label,alpha=.7)
    axes[i,0].set_title(f'{axis}：同方向前进段原始磁场')
    axes[i,1].set_title(f'{axis}：去除约 0.5 秒慢趋势后的波动')
    for ax in axes[i]: ax.set_xlabel('从前进段开始计时 / s'); ax.set_ylabel('磁场 / µT'); ax.grid(alpha=.18); ax.legend(frameon=False)
    axes[i,0].axvspan(0,1,color='grey',alpha=.08); axes[i,0].axvspan(7.54,8.55,color='grey',alpha=.08)
fig.suptitle('布料记录 A / B 对比：均为 80 mm、10 mm/s，细小波动仅比较前进段中部',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(12,6.4),layout='constrained')
for ax,(label,d) in zip(axes,data.items()):
    ax.plot(d['t'],d['raw'][:,2],color=COLORS[label],lw=.8)
    ax.axvline(8.55,color='grey',ls='--',lw=1); ax.set_title(f'{label}：Bz 原始记录，>10,000 µT 相邻跳变 {result[label]["bz_large_jumps"]} 次')
    ax.set_xlabel('记录时间 / s'); ax.set_ylabel('Bz / µT'); ax.grid(alpha=.18)
    ax.text(.98,.94,'虚线前：前进；后：返回\n该通道存在疑似量程/符号回绕，暂不作布料判别依据',ha='right',va='top',transform=ax.transAxes,fontsize=9)
fig.savefig(OUT/'bz_quality.png',dpi=160); plt.close(fig)
result['method']={'primary_phase':'step_001 only','trim_s':1,'residual':'uniform-time interpolation on analysis copy; Savitzky-Golay quadratic 0.5s trend subtraction','inference':'descriptive only; one independent scan per label; no classifier accuracy or independent significance test','bz':'raw retained; no unwrap or sensor-range correction applied'}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for label in RUNS:
    r=result[label]
    print(label,json.dumps({k:r[k] for k in ('n','duration_s','sampling_rate_hz','invalid_rows','gaps_gt_4p5ms','central_n','central_mean_uT','central_std_uT','central_p95_minus_p5_uT','residual_rms_uT','bz_large_jumps','within_record_1s_windows')},ensure_ascii=False))
