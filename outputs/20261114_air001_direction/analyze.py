"""Descriptive comparison of one recorded outward/return pair; no model training."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.signal import welch

SOURCE = Path('/home/serein/classify-cloth/xz-control/data/布料采集数据/空气层_001/空气层_001.csv')
OUT = Path(__file__).resolve().parent
digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))
meta = json.loads((SOURCE.parent / 'metadata.json').read_text())
axes = ['x', 'y', 'z']
labels = ['去程（+X）', '回程（−X）']
colors = ['#1768ac', '#d46b24']
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.unicode_minus': False})

legs = []
stats = []
for stage, name in zip(['step_003', 'step_004'], labels):
    selected = [r for r in rows if r['stage'] == stage]
    time = np.array([int(r['device_time_unwrapped_us']) for r in selected], dtype=float) / 1e6
    time -= time[0]
    delta = np.array([[float(r[f'dB{a}_uT']) for a in axes] for r in selected])
    raw = np.array([[float(r[f'B{a}_uT']) for a in axes] for r in selected])
    valid = np.array([r['sample_valid'] == '1' for r in selected]) & np.isfinite(delta).all(axis=1)
    position = np.array([float(r['motor_x_pulses']) for r in selected]) / meta['context']['config']['x_ppmm']
    assert np.all(np.diff(time) > 0)
    assert len(set(r['device_clock_segment'] for r in selected)) == 1
    baseline = np.array([meta['baseline_B0_uT'][f'B{a}_uT'] for a in axes])
    assert np.allclose(raw - baseline, delta, atol=2e-6)
    leg = dict(time=time, delta=delta, raw=raw, valid=valid, position=position)
    legs.append(leg)
    item = dict(direction=name, stage=stage, rows=len(selected), valid_rows=int(valid.sum()),
                invalid_rows=int((~valid).sum()), duration_s=float(time[-1]),
                mean_sampling_rate_hz=float((len(time)-1)/time[-1]),
                max_timestamp_gap_ms=float(np.diff(time).max()*1000), axes={})
    for k, a in enumerate(axes):
        item['axes'][a] = {}
        for region, mask in [('full', valid), ('trim_1s_each_end', valid & (time >= 1) & (time <= time[-1]-1))]:
            v = delta[mask, k]
            item['axes'][a][region] = dict(n=len(v), mean=float(v.mean()), std=float(v.std(ddof=0)),
                median=float(np.median(v)), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)))
    stats.append(item)

# Position is latest motor software status, not a synchronized encoder measurement.
position_centers = np.arange(.5, 155, 1.)
binned = []
for leg in legs:
    values = np.full((len(position_centers), 3), np.nan)
    for i, center in enumerate(position_centers):
        mask = leg['valid'] & (leg['position'] >= center-.5) & (leg['position'] < center+.5)
        if mask.any():
            values[i] = np.median(leg['delta'][mask], axis=0)
    binned.append(values)

comparison = {}
spectra = []
for leg in legs:
    keep = leg['valid'] & (leg['time'] >= 1) & (leg['time'] <= leg['time'][-1]-1)
    fs = 1000/3
    grid = np.arange(leg['time'][keep][0], leg['time'][keep][-1], 1/fs)
    regular = np.column_stack([np.interp(grid, leg['time'][keep], leg['delta'][keep, k]) for k in range(3)])
    frequencies, psd = welch(regular, fs=fs, nperseg=1024, noverlap=512, axis=0, detrend='constant')
    spectra.append((frequencies, psd))

for k, a in enumerate(axes):
    fstats, rstats = [s['axes'][a]['full'] for s in stats]
    good = np.isfinite(binned[0][:, k]) & np.isfinite(binned[1][:, k])
    good &= (position_centers >= 10) & (position_centers <= 145)
    position_corr = float(np.corrcoef(binned[0][good, k], binned[1][good, k])[0, 1])
    rms = []
    band_fractions = []
    for frequencies, psd in spectra:
        mask = (frequencies >= 1) & (frequencies <= 100)
        power = float(np.trapezoid(psd[mask, k], frequencies[mask]))
        rms.append(float(np.sqrt(power)))
        parts = []
        for lo, hi in [(1, 5), (5, 20), (20, 100)]:
            take = (frequencies >= lo) & (frequencies <= hi)
            parts.append(float(np.trapezoid(psd[take, k], frequencies[take])/power))
        band_fractions.append(parts)
    comparison[a] = dict(mean_difference_return_minus_outward_uT=rstats['mean']-fstats['mean'],
        std_ratio_return_over_outward=rstats['std']/fstats['std'],
        position_binned_median_pearson_r=position_corr,
        spectral_rms_1_to_100_hz_uT=rms,
        spectral_band_fractions_1_5_20_100=band_fractions)

all_bz = np.array([float(r['Bz_uT']) for r in rows])
all_time = np.array([float(r['sensor_t_s']) for r in rows])
jumps = np.flatnonzero(np.abs(np.diff(all_bz)) > 10000) + 1
wrap_period = 65536*.242
wrap = dict(raw_Bz_min_uT=float(all_bz.min()), raw_Bz_max_uT=float(all_bz.max()),
    hypothetical_signed16_boundary_uT=32768*.242, hypothetical_wrap_period_uT=wrap_period,
    jumps_over_10000uT=len(jumps),
    jumps=[dict(seq=int(rows[i]['seq']), stage=rows[i]['stage'], time_s=all_time[i],
        before_uT=all_bz[i-1], after_uT=all_bz[i], jump_uT=all_bz[i]-all_bz[i-1],
        sample_valid=rows[i]['sample_valid']) for i in jumps],
    interpretation='Suspected signed-16-bit range wrap, conditional on the firmware and library settings. Not corrected.')

result = dict(source=str(SOURCE), source_sha256=digest,
    scope='One preselected record: 空气层_001. Descriptive comparison, not an independence or classification test.',
    methods=dict(stats='Recorded-valid samples only; population standard deviation; no denoising.',
        sensitivity='Also recompute after trimming 1 s from each end of each leg.',
        position_alignment='1 mm bins using latest software pulse count; median per bin; correlation over 10–145 mm.',
        spectrum='Middle segment, interpolate at 333.333 Hz, Welch 1024 samples/50% overlap, constant detrend. Bz suspect.'),
    legs=stats, comparison=comparison, bz_diagnostic=wrap)
(OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

fig, panels = plt.subplots(2, 2, figsize=(12.8, 6.8), constrained_layout=True)
for k, a in enumerate(axes[:2]):
    for leg, name, color, values in zip(legs, labels, colors, binned):
        v = leg['delta'][:, k].copy()
        v[~leg['valid']] = np.nan
        panels[k, 0].plot(leg['time'], v, lw=.55, alpha=.8, color=color, label=name, rasterized=True)
        panels[k, 1].plot(position_centers, values[:, k], lw=1.5, color=color, label=name)
    panels[k, 0].set(title=f'ΔB{a}：按各自扫描时间比较', xlabel='单程开始后的时间（s）', ylabel='磁场变化（µT）')
    panels[k, 1].set(title=f'ΔB{a}：按相同 X 位置比较（每 1 mm 中位数）', xlabel='软件位置 X（mm）', ylabel='磁场变化（µT）')
    for panel in panels[k]:
        panel.grid(alpha=.2)
        panel.spines[['top','right']].set_visible(False)
        panel.legend(loc='best', fontsize=9)
fig.suptitle('空气层_001：去程与回程的实测 ΔBx、ΔBy', fontsize=16)
fig.savefig(OUT/'direction_comparison.png', dpi=170)
plt.close(fig)

fig, panels = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for k, a in enumerate(axes[:2]):
    for (freq, psd), name, color in zip(spectra, labels, colors):
        keep = (freq >= 1) & (freq <= 100)
        panels[k].semilogy(freq[keep], psd[keep,k], color=color, label=name, lw=1)
    panels[k].set(title=f'ΔB{a} 波动频谱（去除两端各 1 秒）', xlabel='频率（Hz）', ylabel='功率谱密度（µT²/Hz）')
    panels[k].legend(fontsize=9)
    panels[k].grid(alpha=.2)
panels[2].plot(all_time, all_bz, color='#7155a5', lw=.6, rasterized=True)
for boundary in [-32768*.242,32768*.242]:
    panels[2].axhline(boundary, color='#b83737', ls='--',lw=.9)
panels[2].scatter(all_time[jumps], all_bz[jumps], color='#b83737', s=14, zorder=3)
panels[2].set(title='原始 Bz：14 次 >10,000 µT 突跳', xlabel='整次记录时间（s）', ylabel='原始磁场（µT）')
panels[2].grid(alpha=.2)
fig.savefig(OUT/'spectra_and_bz.png', dpi=170)
plt.close(fig)

lines = ['# 空气层_001 去回程比较', '', f'来源：{SOURCE}', f'SHA256：{digest}', '',
    '单份预先选定记录的描述性分析，不训练分类器，不用时序采样点做独立样本的显著性检验。', '',
    '|通道|去程均值 µT|回程均值 µT|去程标准差 µT|回程标准差 µT|', '|---|---:|---:|---:|---:|']
for a in axes:
    f, r = [s['axes'][a]['full'] for s in stats]
    lines.append(f"|ΔB{a}|{f['mean']:.3f}|{r['mean']:.3f}|{f['std']:.3f}|{r['std']:.3f}|")
lines += ['', '|通道|1–100 Hz 波动 RMS，去程 µT|回程 µT|', '|---|---:|---:|']
for a in axes[:2]:
    f, r = comparison[a]['spectral_rms_1_to_100_hz_uT']
    lines.append(f'|ΔB{a}|{f:.3f}|{r:.3f}|')
lines += ['', '## 方法与边界', '',
    '- 统计只使用 sample_valid=1 且数值有限的点。另计算两端各裁去 1 秒后的结果。',
    '- 位置图按最近电机软件位置分 1 mm 箱取中位数；两路串口无硬件同步，不能解释成精确的同纹理位置配准。',
    '- 频谱仅作补充：按设备时钟插值到 333.333 Hz，Welch 法，1024 点窗，50% 重叠，每窗去均值。',
    '- Bz 原始值靠近 ±7929.856 µT 边界并突跳。配置源码为 GAIN_1X/RES_16；Adafruit 2.0.5 的该配置 Z 换算为 0.242 µT/LSB，65536 个码对应 15859.712 µT。与观察到的突跳相符，高度疑似数值回绕，尚未做硬件验证。',
    '- 原始文件未更改，Bz 未做回绕修正。其巨大差值不能直接认定为布料或运动方向的真实物理差异。',
    '- 可以切成去程和回程两个带方向标签的训练样本；分段有价值不要求统计独立。',
    '- 单份记录不能证明两个单程独立，也不能证明布料可分类。评估新往返时，应按原始往返记录分组。', '',
    '驱动参考：https://github.com/adafruit/Adafruit_MLX90393_Library/blob/2.0.5/Adafruit_MLX90393.cpp',
    '换算表：https://github.com/adafruit/Adafruit_MLX90393_Library/blob/2.0.5/Adafruit_MLX90393.h']
(OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
print(json.dumps(result, ensure_ascii=False, indent=2))
print('OUTPUTS', [p.name for p in OUT.iterdir()])
