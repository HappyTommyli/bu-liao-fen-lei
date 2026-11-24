import json,csv,zipfile
from pathlib import Path
from collections import Counter
DEST=Path(__file__).parent;d=json.loads((DEST/'评估结果.json').read_text())
ss=d['classes'];rr=d['records']
assert len(ss)==15 and len(rr)==535 and sum(s['total'] for s in ss)==535
assert len({r['path'] for r in rr})==535
assert sum(r['eligible'] for r in rr)==sum(s['eligible'] for s in ss)==512
assert Counter(r['status'] for r in rr)==dict(normal=500,suggest_review=12,excluded=10,data_error=13)
assert not d['errors'] or len(d['errors'])==13
assert not d['duplicates'] and not any(x['disabled'] for x in d['batch'].values())
assert sum(s['larger_csv'] for s in ss)==430 and sum(s['larger_support'] for s in ss)==686
assert all(s['larger_support']%2==0 for s in ss)
assert sum(s['validation_support']['cera'] for s in ss)==464
for s in ss:
    assert s['total']==s['eligible']+s['excluded']+s['unknown']+s['data_error']
assert Counter(s['suggestion'] for s in ss)=={'优先暂不纳入':1,'重点复核':2,'保留待验证':8,'优先保留':4}
lines=['# 当前布料采集数据评估','',
'截至本次扫描：15类、535个布料CSV。另10个三角形/矩形轨迹测试不纳入，回收站等43个CSV未纳入。只新增分析报告，未改动原始文件、训练选择、模型或审查规则，未启动训练。','',
'采集质量按当前 admission-v1 / abs-v3 审查：500次通过、12次建议复核且正常权重可训练、10次排除、13次数据错误；合计512次符合训练准入。没有类别触发80%整类停用。','',
'## 按类别统计','',
'分类表现读取两个已完成实验的五折开发集验证预测。前20实验实际入选291 CSV，其中232 CSV进入开发验证；430实验各类数量不均，其中343 CSV进入开发验证。每个CSV去、回两条方向序列保持同组，两个方向不当独立采集。未读取或计算最终测试表现，未对全部535 CSV训练或评价新模型。','',
'|布料|当前CSV|可训练|不可训练|前20 TCN F1|前20 CERA F1|430 TCN F1|430 CERA F1|建议|',
'|---|---:|---:|---:|---:|---:|---:|---:|---|']
for s in ss:
    lines.append(f"|{s['label']}|{s['total']}|{s['eligible']}|{s['total']-s['eligible']}|{s['f1']['tcn']:.1%}|{s['f1']['cera']:.1%}|{s['larger_f1']['tcn']:.1%}|{s['larger_f1']['cera']:.1%}|{s['suggestion']}|")
lines+=['','F1同时考虑误分和漏分，越高越好，不等于识别准确率。统计汇总同一实验每折的验证预测，每条验证方向序列只出现一次。不同实验数据重叠，不是两次独立复验；各类验证支持量见CSV和工作簿。','',
'## 类别筛选建议','',
'1. 优先讨论暂不纳入“印花丝绒”：两实验、两模型共四个F1均低于80%，且在430实验中TCN/CERA分别为57.7%/58.6%。该类35个当前CSV均通过质量审查；难分不表示采集错误。若只暂不纳入这一类，剩14类、477个可训练CSV。',
'2. 空气层、长毛格呢列为第二批重点复核：430实验两模型均低于80%；长毛格呢在TCN与CERA之间差距很大，是否保留取决于主模型与新数据结果。若连同印花丝绒共暂不纳入这三类，剩12类、409个可训练CSV。',
'3. 亮丝粗花呢、仿皮毛一体、岩石肌理面料、织棉提花缎在两实验两模型中F1均至少85%，优先保留。其余8类保留待验证，不能凭前20实验低分贸然删除；例如珠地覆膜430实验TCN达91.1%，大衣呢达89.9%。',
'4. 80%/85%是本次探索筛选参考线，不是原有训练准入规则或统计显著性界限。430实验各类仅17～45个入选CSV，且类别数量不均；两实验不能据此认定增加样本必然提升某类。',
'5. 不挑出“模型分错的单条CSV”删除。若改变目标类别，应作为新的分类任务重新训练，并用未参与筛选的新独立采集验证，不能把删类后的指标直接当原15类任务的提升。','',
'## 不准入的23次采集','',
'以下是现行规则审查建议，并未实际移动或删除文件。去回程一起不准入。','',
'|布料|记录|状态|原因|','|---|---|---|---|']
for r in rr:
    if not r['eligible']:lines.append('|'+r['label']+'|'+r['record']+'|'+r['status']+'|'+'；'.join(r['reasons'])+'|')
lines+=['','仿皮毛一体的13次数据错误集中在021～028、030、032～035；接触范围内step_004有效采样间隔超过20ms，按现行协议不准入。这类历史开发集表现为100% F1，不能因此移除整个布料类别，建议复采这些记录。','',
'## 波形统计解释','',
'幅值统计仅使用当前可训练CSV。每个CSV的代表幅值为去回有效段|ΔBy|各自中位数的平均。变异系数是各CSV代表幅值的样本标准差/均值，反映采集之间幅值波动，不是所有时序采样点噪声。去回差异率使用尚未标准化的共同X范围中位数，按现行四项规则共同判定，不能只看10%一条。',
'完整固定接触窗：去程45～142mm、回程16～114mm；保留原有0.2秒尾部裕量与333Hz预处理。仅评价ΔBy主实验，不据此断言加入其他通道后仍无法区分。','',
'## 来源','',d['scope']['source'],'',d['scope']['model_source']+'/metadata.json','',d['larger_model']['task']+'/metadata.json','',
'所有可读取原记录及430实验快照源均核对CSV和metadata指纹。审查规则版本：'+d['scope']['review_rule']+'。每条记录的指纹、路径、指标、审查原因见逐次采集明细。']
(DEST/'布料数据评估报告.md').write_text('\n'.join(lines))
for fn,expected in [('布料评估统计.csv',16),('逐次采集审查明细.csv',536)]:
    with (DEST/fn).open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.reader(f)))==expected
with zipfile.ZipFile(DEST/'布料数据评估统计.xlsx') as z:
    assert z.testzip() is None
    assert sum(n.startswith('xl/worksheets/sheet') and n.endswith('.xml') for n in z.namelist())==2
print('VERIFIED: 535 CSV, 15 classes, 512 eligible, 10 excluded, 13 errors, 12 eligible reviews; workbook + CSV row counts correct.')
print('CANDIDATES',[(s['label'],s['suggestion']) for s in ss])
