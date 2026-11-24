import json,csv,collections,itertools
from pathlib import Path
import numpy as np
DEST=Path(__file__).parent;d=json.loads((DEST/'质量与去回一致性结果.json').read_text());rows=d['records'];classes=d['classes'];inputs=json.loads((DEST/'表格输入.json').read_text())
def compact(records):
    nums=sorted(int(x.rsplit('_',1)[1])for x in records);out=[]
    for _,group in itertools.groupby(enumerate(nums),lambda z:z[1]-z[0]):
        a=[v for _,v in group];out.append(f'{a[0]:03d}'if len(a)==1 else f'{a[0]:03d}～{a[-1]:03d}')
    return '、'.join(out)
for s,row in zip(classes,inputs['summaryrows']):row[-2:]=[compact(s['bad_records']),compact(s['hold_records'])]
inputs['summaryheaders'][-2:]=['不合格编号','暂缓编号']
(DEST/'表格输入.json').write_text(json.dumps(inputs,ensure_ascii=False))
with(DEST/'按布料质量汇总.csv').open('w',encoding='utf-8-sig',newline='')as f:w=csv.writer(f);w.writerow(inputs['summaryheaders']);w.writerows(inputs['summaryrows'])
bad=[r for r in rows if r['action'].startswith('不合格')];held=[r for r in rows if r['action'].startswith('暂缓')];reviews=[r for r in rows if r['action']=='可用·建议复核'];usable=[r for r in rows if r['action'].startswith('可用')]
rates=np.array([r['direction']['common_rate']for r in usable if r.get('direction',{}).get('status')=='assessed'])
gaps=[max(s.get('max_gap_ms')or 0 for s in r['stage_details'].values())for r in bad if r['status']=='data_error']
offsets=collections.Counter(round(r['coordinate_offset_candidate_mm'],2)for r in held if'coordinate_offset_candidate_mm'in r)
aligned_gap=[r for r in held if any((v or 0)>20.00001for v in r.get('conditional_aligned_max_gap_ms',{}).values())]
stats=dict(total=len(rows),classes=len(classes),passed=sum(r['action']=='可用·通过'for r in rows),review=len(reviews),usable=len(usable),bad=len(bad),held=len(held),
    direction_median=float(np.median(rates)),direction_p90=float(np.quantile(rates,.9)),direction_max=float(rates.max()),gap_bad_min_ms=float(min(gaps)),gap_bad_max_ms=float(max(gaps)),
    historical_bad=sum(r.get('history')=='excluded'for r in bad),new_four_condition_bad=sum(r['status']=='excluded'and r.get('history')!='excluded'for r in bad),
    sampling_gap_bad=sum(r['status']=='data_error'for r in bad),offset_counts=dict(offsets),aligned_gap_count=len(aligned_gap),
    supplemental_xyz_alert_records=sum(bool(r.get('supplement_alerts'))for r in rows),supplement_failures=sum(bool(r.get('supplement_failure'))for r in rows))
d['statistics']=stats
(DEST/'质量与去回一致性结果.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
lines=['# 当前布料采集质量与去回一致性审查','',
f"审查范围固定于北京时间{d['scope']['started_at']}；18类、{len(rows)}个布料CSV。另10个轨迹测试与回收站等目录未混入布料统计。分析过程中继续新增的文件不自动计入。",'',
'仅检查原始采集质量，不使用模型分数、类别重合度或分类难度。原始文件、应用代码、训练选择和旧模型均未更改。审查使用现行admission-v1与默认abs-v3协议，并对全部候选重新读取原始CSV。','',
'## 结果','',
f"- 可用：{len(usable)}条，其中通过{stats['passed']}条、建议复核{len(reviews)}条（仍正常权重可用）。",
f"- 按现行规则建议排除：{len(bad)}条，其中历史确认排除{stats['historical_bad']}条、另有{stats['new_four_condition_bad']}条满足去回不一致四条件、{stats['sampling_gap_bad']}条存在接触窗有效采样缺口。",
f"- 暂缓核对坐标：{len(held)}条。脚本目前将这些条目标为数据错误；分析将它们单列为协议/坐标不兼容，不能直接说磁测量已损坏。",
f"- 可用记录共同X范围去回幅值差异率：中位数{stats['direction_median']:.2%}，90分位{stats['direction_p90']:.2%}，最大{stats['direction_max']:.2%}。",'',
'|布料|CSV|通过|可用建议复核|不合格|坐标暂缓|可用去回差异中位数|','|---|---:|---:|---:|---:|---:|---:|']
for s in classes:
    med=f"{s['usable_rate_median']:.2%}"if s['usable_rate_median']is not None else'无法依旧协议判定'
    lines.append(f"|{s['label']}|{s['total']}|{s['passed']}|{s['review']}|{s['bad']}|{s['hold']}|{med}|")
lines+=['','## 建议先排除的38条','',
'下表每个编号表示一个往返CSV，去回一起不纳入。所有原始文件仍保留。','',
'|布料|编号|原因|','|---|---|---|']
for label in sorted({r['label']for r in bad}):
    for reason in ('历史确认排除','四项去回不一致均满足','接触窗采样缺口超过20ms'):
        subset=[r for r in bad if r['label']==label and(reason=='历史确认排除'and r.get('history')=='excluded'or reason=='四项去回不一致均满足'and r['status']=='excluded'and r.get('history')!='excluded'or reason=='接触窗采样缺口超过20ms'and r['status']=='data_error')]
        if subset:lines.append(f"|{label}|{compact([r['record']for r in subset])}|{reason}|")
lines+=['',f"28条采样缺口记录的最大缺口约{min(gaps):.0f}～{max(gaps):.0f}ms。它们按当前20ms上限不合格；这是现行协议门槛，不代表21ms就使整段物理信息完全失效。本分析未放宽该门槛，也未插补缺口。缺口次数、起止时间及X位置见逐条结果。",'',
'## 135条坐标范围不兼容','',
f"检测到软件X范围整体平移：{offsets.get(-24.35,0)}条约为−24.35～130.65mm，{offsets.get(26.0,0)}条约为26～181mm。两方向的记录行程仍约155mm，与recipe的±155mm相符。旧协议要求去程45～142mm、回程16～114mm，因此前一种缺去程窗口高端，后一种缺回程窗口低端。",'',
'metadata记录了人工机械基准和起始位置，但仅凭CSV与metadata不能证明物理起点/平台位置没有改变。观察到的是软件坐标范围变化，不能擅自认定只需减去一个常数即可修复。建议先核对机械基准、采集起点和CSV坐标的关系；暂不与旧坐标数据直接混训。','',
'|布料|暂缓编号|','|---|---|']
for label in sorted({r['label']for r in held}):lines.append(f"|{label}|{compact([r['record']for r in held if r['label']==label])}|")
lines+=['',
f"补充做了“假设仅为零点平移”的条件性计算：135条均未触发去回幅值四条件；其中126条未见平移后旧接触窗内超过20ms的采样缺口，另{len(aligned_gap)}条仍存在缺口。该计算只说明核对坐标可能避免不必要的复采，不代表135条已经通过完整质量审查，也没有改变任何文件坐标。",'',
'平移假设下仍需重点检查：'+ '、'.join(r['record']for r in aligned_gap)+'。','',
'## 建议复核但保持可用的12条','',
'历史11条边界复核记录保持正常权重；新增长毛格呢023仅完整范围差异超过10%，共同范围未达到10%，不满足全部异常条件。','',
'|记录|完整窗差异率|共同X差异率|四条件均满足|','|---|---:|---:|---|']
for r in reviews:
    dm=r['direction'];lines.append(f"|{r['record']}|{dm['full_rate']:.2%}|{dm['common_rate']:.2%}|否|")
lines+=['','## 检查方法与限制','',
'1. 使用现有数据解析器检查必要列、可追溯悬空基线、By差分、固定接触窗覆盖、333Hz时基、时间递增/时钟分段、20ms缺口、扫描速度6mm/s及其10%容差、方向和已记录电机故障。By差分与原By−基线容差0.01μT，范围边界容差2mm。',
'2. 补查三轴原始值与差分/基线关系，并测量各方向有效点数、无效点数、最大缺口和位置。没有记录的撞击、离面、打滑不能据此证明未发生。',
'3. 去回一致性使用未标准化|ΔBy|，共同X范围按有效原采样值中位数；位置箱宽2mm，每方向至少3点，至少10个共同箱。差异率=两方向中位数绝对差/两者均值。',
'4. 自动去回排除必须同时满足：完整窗及共同窗差异率均≥10%；偏差同向；共同窗幅值差≥max(5μT,3倍基线By噪声标准差)；至少80%位置箱同向偏差。形状相关、相对RMS另外展示，不新增形状阈值排除。',
'5. 对本就未通过完整性检查的文件，补算去回指标只描述已记录部分；不能借此冒充完整可用。坐标不兼容文件的原窗口指标尤其不能当作真实有效接触段结论。',
'6. 历史7条按CSV与metadata双指纹识别；同名新采集重新审查。本批次没有类别触发现行至少10个可评估独立组且80%大幅异常的整类停用条件。',
'7. 当前38条明确不合格与135条坐标暂缓，合计173条均不能直接按当前旧协议进入训练。后者在软件里仍是原来的数据错误状态，本报告没有替软件解除准入限制。','',
f"三轴补查异常记录数：{stats['supplemental_xyz_alert_records']}；补查未完成数：{stats['supplement_failures']}。",'',
'## 来源与追溯','',d['scope']['source'],'',
'完整重审批次：'+str(DEST/'当前完整审查批次.json'),'',
'每条CSV路径、CSV和metadata指纹、触发原因及定量指标均保存在质量明细中；判定规则全文和参数保存在本次JSON审查批次中。']
(DEST/'采集质量与去回一致性报告.md').write_text('\n'.join(lines))
assert len(rows)==726 and len(classes)==18 and len(usable)==553 and len(bad)==38 and len(held)==135 and len(reviews)==12
assert sum(s['total']for s in classes)==len(rows)
assert sum(d['counts'].values())==len(rows)
assert stats['historical_bad']==7 and stats['new_four_condition_bad']==3 and stats['sampling_gap_bad']==28
assert stats['supplemental_xyz_alert_records']==stats['supplement_failures']==0
assert not d['scope']['model_results_used']and not d['scope']['raw_sources_modified']
print('VERIFIED',stats)
