import json,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent;a=json.loads((OUT/'audit.json').read_text());rs=a['records']
def status(d):return d.get('quality') if d['status']=='loaded' else d['status']
names={'normal':'通过','suggest_review':'建议复核','excluded':'排除/建议复采','undetermined':'待复核','error':'数据错误','unassessable':'依据不足'}
text=['# 布料采集数据新：仅By质量检查','',f"审查完成时间：{a.get('refreshed_at','')}。共{len(rs)}个CSV。",'', '固定裁剪规则：去程51～140 mm，回程30～120 mm；信号为|dBy|。相对接触段规则另列对照。','', '本次只在审查内存中重新计算By质量：仅恢复原因为isolated_spike且By自身有限、无同类尖峰的点；保留其他无效原因，并重新检查By尖峰。CSV、metadata、训练脚本均未修改。','', '通过表示满足当前数据与质量规则，不能证明不存在机械滑动等无法从文件验证的问题。','']
for rule in ('fixed_51_140','relative_035'):
 text.extend([f'## {rule}','', '| 模式 | 结果 |','|---|---|'])
 for mode in ('original','by_only'):
  c=collections.Counter(status(r['checks'][rule][mode]) for r in rs)
  text.append('| '+mode+' | '+'；'.join(names.get(k,k)+str(v)+'条' for k,v in c.items())+' |')
 text.extend(['','### 仅By仍有问题的记录','','| 文件 | 状态 | 说明 |','|---|---|---|'])
 for r in rs:
  d=r['checks'][rule]['by_only'];s=status(d)
  if s=='normal':continue
  note=d.get('error') or '；'.join(d.get('notes',[]));text.append(f"| [{r['name']}]({r['path']}) | {names.get(s,s)} | {note}；采集状态={r['outcome']} |")
 text.append('')
text.extend(['## 全部记录','','| 文件 | 日期UTC | 固定规则 | 相对规则 | 原无效点 | 恢复的其他通道尖峰点 | By尖峰 |','|---|---|---|---|---:|---:|---:|'])
for r in rs:
 q=r['quality_mask'];text.append(f"| {r['name']} | {r['created_at']} | {names.get(status(r['checks']['fixed_51_140']['by_only']))} | {names.get(status(r['checks']['relative_035']['by_only']))} | {q['original_invalid']} | {q['restored_other_axis_spike']} | {q['by_isolated_spike']} |")
(OUT/'审查报告.md').write_text('\n'.join(text)+'\n')
for rule in ('fixed_51_140','relative_035'):
 print(rule,'original',dict(collections.Counter(status(r['checks'][rule]['original']) for r in rs)),'by_only',dict(collections.Counter(status(r['checks'][rule]['by_only']) for r in rs)))
 for r in rs:
  d=r['checks'][rule]['by_only']
  if status(d)!='normal':print(r['name'],status(d),d.get('error'),d.get('notes'),r['outcome'])
print('mask_totals',{k:sum(r['quality_mask'][k] for r in rs) for k in rs[0]['quality_mask']})
print('source_stable_during_checks',all(r['unchanged_after_check'] for r in rs))
print('TOTAL',len(rs),'labels',len(set(r['label'] for r in rs)))
