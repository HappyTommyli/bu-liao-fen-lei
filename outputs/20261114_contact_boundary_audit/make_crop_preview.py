"""Export a reviewable, conservative crop proposal from the existing boundary audit.

This is an analysis export, not the final training preprocessing implementation.
Source CSV values are preserved; the retained boundaries remain provisional.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
AUDIT=HERE/'results.json'
OUT=HERE.parent/'20261114_contact_crop_preview_v1'
ENTRY_GUARD_S=3.5
EXIT_GUARD_S=.5

def main():
    audit=json.loads(AUDIT.read_text())
    if OUT.exists():
        raise FileExistsError(f'Preview output already exists; preserve it and choose another output: {OUT}')
    source_hashes={v['path']:v['sha256'] for v in audit['sources']}
    proposed=[]
    chart_data={}
    # Read and verify all sources before creating derived files.
    by_source={}
    for item in audit['records']:
        by_source.setdefault(item['source'],[]).append(item)
    for source_name,items in by_source.items():
        source=Path(source_name)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hashes[source_name]
        with source.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);headers=reader.fieldnames;all_rows=list(reader)
        if {item['stage'] for item in items}!={'step_003','step_004'}:
            raise ValueError(f'Incomplete pair: {source}')
        pair=dict(run=items[0]['run'],label=items[0]['label'],source=source_name,
                  source_sha256=source_hashes[source_name],status='provisional_needs_boundary_review',segments=[])
        exports=[]
        for item in sorted(items,key=lambda v:v['stage']):
            raw=[v for v in all_rows if v['stage']==item['stage']]
            time=np.array([int(v['device_time_unwrapped_us']) for v in raw],dtype=float)/1e6
            time-=time[0]
            candidate=item['xy_candidate']
            if candidate is None:raise ValueError(f'No candidate: {source}, {item["stage"]}')
            start=candidate['start_s']+ENTRY_GUARD_S
            end=candidate['end_s']-EXIT_GUARD_S
            keep=(time>=start)&(time<=end)
            selected=[v for v,ok in zip(raw,keep) if ok]
            if not selected or time[keep][-1]-time[keep][0]<5:
                raise ValueError(f'Insufficient retained interval: {source}, {item["stage"]}')
            suffix='outward' if item['stage']=='step_003' else 'return'
            csv_relative=f'cropped/{item["run"]}/{item["run"]}_{suffix}.csv'
            segment=dict(direction=item['direction'],stage=item['stage'],csv=csv_relative,
                candidate_start_s=candidate['start_s'],candidate_end_s=candidate['end_s'],
                requested_start_s=start,requested_end_s=end,
                retained_start_s=float(time[keep][0]),retained_end_s=float(time[keep][-1]),
                retained_duration_s=float(time[keep][-1]-time[keep][0]),
                first_seq=int(selected[0]['seq']),last_seq=int(selected[-1]['seq']),
                rows=len(selected),invalid_rows=sum(v['sample_valid']!='1' for v in selected))
            pair['segments'].append(segment)
            exports.append((csv_relative,headers,selected))
            xy=np.array([np.hypot(float(v['dBx_uT']),float(v['dBy_uT'])) for v in raw])
            chart_data[(item['run'],item['direction'])]=(time,xy,segment)
        proposed.append((pair,exports))

    OUT.mkdir(parents=True)
    manifest=dict(status='provisional_needs_boundary_review',
        method='Use each audit XY candidate; remove 3.5 s after its entry and 0.5 s before its exit. Same guards for all cloths and both directions.',
        entry_guard_s=ENTRY_GUARD_S,exit_guard_s=EXIT_GUARD_S,
        assumptions='Audit assumes first/last 1 s of each leg airborne. Fixed guard is conservative and tied to the current 155 mm, 6 mm/s acquisition protocol, not proof of settling.',
        raw_data_policy='Copy original fields and values. No resampling, no smoothing of exports, no Bz correction. Invalid flags retained for later quality handling.',
        pairing='Each outward and return segment belongs to the same run; combine their features into one training sample.',
        pairs=[])
    for pair,exports in proposed:
        for relative,headers,selected in exports:
            target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('x',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(selected)
        meta=OUT/'cropped'/pair['run']/'segments.json'
        meta.write_text(json.dumps(pair,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest['pairs'].append(pair)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'segment_summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['run','label','direction','stage','retained_start_s','retained_end_s','retained_duration_s','rows','invalid_rows','first_seq','last_seq','csv']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for pair in manifest['pairs']:
            for segment in pair['segments']:
                writer.writerow({k:pair[k] if k in ('run','label') else segment[k] for k in fields})

    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False})
    def draw(runs,filename):
        fig,axs=plt.subplots(len(runs),2,figsize=(12.6,2.6*len(runs)+1),squeeze=False,constrained_layout=True)
        for i,run in enumerate(runs):
            for j,direction in enumerate(['去程','回程']):
                time,xy,segment=chart_data[(run,direction)];ax=axs[i,j]
                ax.plot(time,xy,color='#1768ac',lw=.6,rasterized=True)
                ax.axvspan(segment['candidate_start_s'],segment['candidate_end_s'],color='#888888',alpha=.15,label='候选接触区间')
                ax.axvspan(segment['retained_start_s'],segment['retained_end_s'],color='#2e9b60',alpha=.22,label='本版保留区间')
                for edge in ('retained_start_s','retained_end_s'):ax.axvline(segment[edge],color='#25834e',lw=1,ls='--')
                ax.set(title=f"{run} · {direction}：保留 {segment['retained_start_s']:.2f}–{segment['retained_end_s']:.2f} s",
                    xlabel='该 step 开始后时间（s）',ylabel='XY 模长（µT）')
                ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(loc='best',fontsize=8)
        fig.suptitle('自动裁剪预览：step_003 去程，step_004 回程\n候选起点后去掉 3.5 秒、终点前去掉 0.5 秒；边界尚待核对',fontsize=13)
        fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    draw(['空气层_001'],'air001_crop_preview.png')
    draw(['空气层_001','织棉提花缎_001','岩石肌理面料_001'],'three_cloths_crop_preview.png')
    lines=['# 布料往返接触段裁剪预览','',
        '25 次往返，各导出一个去程和一个回程 CSV，共 50 个单程文件。它们仍对应 25 个配对训练样本。','',
        '这是供边界核对的第一版分析导出，尚不是经过人工标注验证的正式接触段。','',
        '规则：stage=step_003 为去程，step_004 为回程。以先前 XY 响应候选区间为基础，接触候选起点后裁去 3.5 秒，候选终点前裁去 0.5 秒。所有类别、两个方向使用相同保护时间。',
        '', '3.5 秒是按当前数据中上平台过渡采用的保守保护时间，不是通过接触传感器测得的稳定时刻。换速度或改变路径需重新检查。',
        '', 'CSV 保留原列、原值、原始时间和 seq；有效性标记保留。没有修改原始数据、插值、修改 Bz 或进行幅值归一化。',
        '', '每条边界、源文件 SHA256、输出路径与往返配对关系记录在 manifest.json 和 segment_summary.csv。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for source,digest in source_hashes.items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    durations=[s['retained_duration_s'] for p in manifest['pairs'] for s in p['segments']]
    print(json.dumps(dict(output=str(OUT),pairs=len(proposed),segments=len(durations),duration_range_s=[min(durations),max(durations)],
        example=next(p for p in manifest['pairs'] if p['run']=='空气层_001')),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
