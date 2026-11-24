"""Explicit selection sets from raw overlap only. No training outcomes."""
import json,csv,itertools
from pathlib import Path
import numpy as np
DEST=Path(__file__).parent;d=json.loads((DEST/'重合度分析.json').read_text());src=json.loads((DEST/'原始数据清单.json').read_text())
classes=d['classes'];labels=[s['label'] for s in classes]
pairmap={frozenset((p['a'],p['b'])):p for p in d['pairs']}
def choose(limit,ratio):
    edges=[(labels.index(p['a']),labels.index(p['b'])) for p in d['pairs'] if p['overlap']>limit or p['separation']<ratio]
    best=(-1,-1e10);selected=[]
    for mask in range(1,1<<len(labels)):
        n=mask.bit_count()
        if n<best[0] or any((mask>>i)&(mask>>j)&1 for i,j in edges):continue
        ix=[i for i in range(len(labels)) if mask>>i&1];key=(n,-sum(classes[i]['cv'] for i in ix))
        if key>best:best=key;selected=ix
    out=[labels[i]for i in selected]
    metrics=[pairmap[frozenset((a,b))]for a,b in itertools.combinations(out,2)]
    return dict(labels=out,overlap_limit=limit,minimum_ratio=ratio,max_overlap=max(p['overlap']for p in metrics),min_ratio=min(p['separation']for p in metrics),
        eligible=sum(s['eligible']for s in classes if s['label']in out))
strict=choose(.20,2.);moderate=choose(.50,1.)
for s in classes:
    s['strict']='保留' if s['label']in strict['labels'] else '换布候选'
    s['moderate']='保留' if s['label']in moderate['labels'] else '换布候选'
    s['closest_texture']=pairmap[frozenset((s['label'],s['closest']))]['texture_overlap']
d['selection']=dict(strict=strict,moderate=moderate,tie_break='Maximum class count; then minimum sum of per-class amplitude CV. Other maximum-size sets may exist.')
classes.sort(key=lambda s:(s['strict']!='保留',s['max_overlap']))
headers=['布料','本次CSV','可用CSV','幅值中位数μT','同类幅值CV','最相似布料','最高波形带重合率','与该类距散比','严格方案','五类折中方案','全部高重合对象']
rows=[[s['label'],s['total'],s['eligible'],s['amplitude'],s['cv'],s['closest'],s['max_overlap'],s['closest_separation'],s['strict'],s['moderate'],'；'.join(s['high_wave_partners'])or'无']for s in classes]
pairheaders=['布料A','布料B','波形带重合率','类间RMS距离μT','A类内RMSμT','B类内RMSμT','距散比','去程重合率','回程重合率','去趋势RMS区间重合率','相邻差分区间重合率','1至20Hz功率区间重合率','20至100Hz功率区间重合率','共同X重合率','可比较编号段数','编号段平均重合率','波形和四项纹理均重合']
pairrows=[[p['a'],p['b'],p['overlap'],p['distance'],p['scatter_a'],p['scatter_b'],p['separation'],p['out_overlap'],p['back_overlap'],*p['texture_overlap'],p['common']['overlap'],len(p['cohorts']),float(np.mean([c['overlap']for c in p['cohorts']])) if p['cohorts']else None,'是'if p['hard_pair']else'否']for p in d['pairs']]
recordheaders=['布料','采集记录','状态','本次用于重合分析','代表幅值μT','去趋势RMSμT','相邻差分均值μT','1至20Hz功率μT²','20至100Hz功率μT²','原因','CSV指纹','metadata指纹','CSV路径']
recordrows=[[r['label'],r['record'],r['status'],'是'if r['eligible']else'否',r.get('amplitude'),r.get('residual_rms'),r.get('mean_abs_diff'),r.get('power_1_20'),r.get('power_20_100'),'；'.join(r['reasons']),r['sha256'],r['metadata_sha256'],r['path']]for r in src['records']]
for fn,hh,rr in [('原始数据布料去留统计.csv',headers,rows),('布料两两重合统计.csv',pairheaders,pairrows),('本次采集分析明细.csv',recordheaders,recordrows)]:
    with (DEST/fn).open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.writer(f);w.writerow(hh);w.writerows(rr)
(DEST/'重合度分析.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
(DEST/'表格输入.json').write_text(json.dumps(dict(headers=headers,rows=rows,pairheaders=pairheaders,pairrows=pairrows,recordheaders=recordheaders,recordrows=recordrows),ensure_ascii=False))
lines=['# 基于原始磁数据的布料筛选','',
f"本次固定扫描 {src['scope']['csv_count']} 个布料CSV、15类；{src['scope']['eligible']} 个可用CSV参与原始信号比较。另10个轨迹测试不纳入。审查状态：{src['quality']}。采集中或读取中变化的记录列为pending，未纳入；分析完成后的新增/替换不自动刷新本报告。",'',
'本报告没有读取模型权重、预测、训练分数或训练报告。仅复用CSV和metadata两个指纹都完全一致的历史完整性/质量审查结果；同名新文件重新审查。原始文件、模型和训练选择均未更改。','',
'## 如何解释指标','',
'- 保留|ΔBy|幅值。去、回程分别按软件X位置统计2mm箱内有效原始采样中位数。X位置约1.2mm才更新一次，故不使用1mm空箱制造伪分辨率。统一比较窗为去程45～141mm、回程17～113mm，较现行固定接触窗各收去不足一个2mm箱的边缘。',
'- 每类每个位置箱取不同CSV的第5、50、95百分位。两类波形带重合率为所有位置箱区间交集宽度之和/并集宽度之和，去回各48箱。它是点位波动带的面积交并比，既不是样本重复比例，也不是错分率或概率。90%带是经验百分位范围，不是置信区间。',
'- 类间距离D为两个类中位波形的RMS差。类内波动W为每条CSV波形到该类中位波形的RMS距离，再取中位数。距散比=D/(W_A+W_B)，越高表示类间差异相对类内变化越明显。',
'- 同类CV是每个CSV去回幅值中位数均值的样本标准差/均值，不能替代波形差异。',
'- 纹理检查保留333Hz时基，计算去趋势残差RMS、相邻采样绝对差均值、1～20Hz与20～100Hz功率；去趋势用167点约0.5s滑动均值，功率使用Welch/1024点。这些是原始信号统计，不是训练。纹理也可能包含传感器噪声、接触和机械运动影响，不能单凭它断言材料机制。',
'- 波形高重合参考：重合率≥50%且距散比<1.5。纹理也相近参考：上述四个指标各自90%区间交并比均≥30%。这些是本次探索性工程筛选阈值，不是文献公认界限或训练保证。','',
'## 保留组合与换布候选','',
'去留是相对于“同时保留哪些类别”的组合判断，不是材料优劣的绝对排名。每套方案枚举15类全部组合，先取满足两两阈值的最大类别数，再选同类CV总和最小的组合；这只是明确的默认取舍，可能有其他同样规模的可行组合。算法只计算距离和区间，不拟合分类器。','',
f"严格易分方案：所有保留类别之间波形重合率≤20%、距散比≥2。选出{len(strict['labels'])}类：{'、'.join(strict['labels'])}；可用CSV {strict['eligible']}；实际最大重合率{strict['max_overlap']:.1%}，最小距散比{strict['min_ratio']:.2f}。这适合用户优先依靠大幅值/粗波形差异选布，不依赖微小纹理的目标；不表示其余布料无法分类。",'',
f"五类折中方案：重合率≤50%、距散比≥1。选出{len(moderate['labels'])}类：{'、'.join(moderate['labels'])}；可用CSV {moderate['eligible']}；实际最大重合率{moderate['max_overlap']:.1%}、最小距散比{moderate['min_ratio']:.2f}。类别更多，边界更接近，不能称为严格明显分离。",'',
'|布料|本次CSV|可用|同类CV|最相似对象|最高波形带重合率|距散比|严格方案|五类方案|',
'|---|---:|---:|---:|---|---:|---:|---|---|']
for s in classes:lines.append(f"|{s['label']}|{s['total']}|{s['eligible']}|{s['cv']:.1%}|{s['closest']}|{s['max_overlap']:.1%}|{s['closest_separation']:.2f}|{s['strict']}|{s['moderate']}|")
lines+=['','“最高重合”和“最相似对象”基于完整15类；方案中的某个保留类可以与已移出的类别高度重合，这并不违反该方案保留集内的两两阈值。','',
'## 不应直接归因为布料本身的现象','',
'汇总所有记录后，亮丝欧根纱/覆膜帆布重合78.9%；按001～010、011～020、021～030、031～035编号段分别比较约21%、22%、32%、9%。编号段只是敏感性检查，不能冒充已证实的独立采集批次。该变化说明跨编号段幅值漂移明显放大总体重合。若未来接触条件/基线一致性得到改善，部分换布候选可能重新具备保留价值。',
'多数高波形重合配对仍有部分纹理统计差异。印花丝绒/珠地覆膜、印花丝绒/大衣呢同时触发波形与四项纹理区间重合规则，值得优先考虑替换或做专门重复采集；但它们分编号段时没有持续达到50%波形重合，仍不能断言差异完全来自材料自身。',
'如果目标是尽量避开细微差异，严格三类可作为选新布的参照：新布先与打算保留的全部布料比较，不能只和某一类差别大就加入。当前单通道的筛选结论不代表Bx/Bz或其他传感方式也分不开。','',
'## 采集问题单列','',
'采样缺口、已确认去回不一致、未完成采集只影响相应记录的资格，不等于换掉整个布料。详细状态和来源指纹见采集明细。当前gap阈值仍为20ms，未为本分析放宽协议。','',
'## 来源','',str(src['scope']['source']),'',
'质量审查指纹缓存：'+str(DEST.parent/'2026-11-18-current-evaluation/逐次采集审查明细.csv'),'',
'统计缓存：'+str(DEST/'原始波形统计缓存.npz')]
(DEST/'原始数据布料筛选报告.md').write_text('\n'.join(lines))
print('STRICT',strict);print('MODERATE',moderate)
assert len(classes)==15 and len(d['pairs'])==105 and len(src['records'])==src['scope']['csv_count']
assert sum(s['eligible']for s in classes)==src['scope']['eligible']
assert not d['scope']['model_results_used']
for r in d['pairs']:assert 0<=r['overlap']<=1 and np.isfinite(r['separation'])
assert strict['max_overlap']<=.20 and strict['min_ratio']>=2
assert moderate['max_overlap']<=.50 and moderate['min_ratio']>=1
print('COUNTS AND RAW-ONLY SELECTION VERIFIED')
