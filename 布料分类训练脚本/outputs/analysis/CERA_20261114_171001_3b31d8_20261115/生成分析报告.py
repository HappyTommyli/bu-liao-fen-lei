"""Build reproducible tables and static figures from saved results and frozen audit."""
import csv,json,math,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
ROOT=Path('/home/serein/classify-cloth/布料分类训练脚本')
RUN=ROOT/'outputs/models/CERA_20261114_171001_3b31d8';OUT=Path(__file__).parent
meta=json.loads((RUN/'metadata.json').read_text());audit=json.loads((OUT/'冻结模型复核.json').read_text())
keys=['by_full','bxby_full','xy_full'];names=['ΔBy 单通道','ΔBx＋ΔBy 双通道','XY 平面模长'];colors=['#2474B7','#D26927','#638D49'];labels=meta['labels'];qs=np.array([10,20,30,50,75,100])
rs={k:json.loads((RUN/k/'result.json').read_text()) for k in keys}
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(14,5.4))
for k,n,c in zip(keys,names,colors):
 y=[100*x['accuracy'] for x in rs[k]['prefix_metrics']];axes[0].plot(qs,y,'o-',color=c,label=n,lw=2)
 for i,(x,v) in enumerate(zip(qs,y)):
  dy=10 if k=='bxby_full' else ((-18 if i in (0,2) else 10) if k=='by_full' else (10 if i in (0,2) else -18))
  axes[0].annotate(f'{v:.1f}',(x,v),xytext=((-20,0) if k=='xy_full' and i==5 else (0,dy)),textcoords='offset points',ha='center',fontsize=9,color=c)
axes[0].axhline(100/13,c='gray',ls=':',label='均匀随机猜测期望 7.69%');axes[0].set(xlabel='单程裁剪段扫描进度（%）',ylabel='独立测试集准确率（%）',ylim=(0,95),title='同一批 52 个测试单程：中途与完整扫描');axes[0].set_xticks(qs);axes[0].grid(alpha=.15);axes[0].legend(fontsize=9)
for i,(part,n,c) in enumerate([('train','训练集','#85AFC7'),('val','验证集','#87AC7A'),('test','测试集','#D18354')]):
 y=[audit['models'][k]['partitions'][part]['metrics'][-1]['accuracy']*100 for k in keys];x=np.arange(3)+(i-1)*.24;axes[1].bar(x,y,width=.23,label=n,color=c)
 for xx,yy in zip(x,y):axes[1].text(xx,yy+1,f'{yy:.1f}',ha='center',fontsize=9)
axes[1].set_xticks(range(3),names);axes[1].set(ylabel='完整扫描准确率（%）',ylim=(0,100),title='冻结的最佳模型重新评估（均关闭 dropout）');axes[1].legend();axes[1].grid(axis='y',alpha=.15)
fig.suptitle('13 类布料 · 130 CSV · CERA-Net 正式训练结果',fontsize=17);fig.tight_layout();save(fig,'01_总体表现与扫描进度')
fig,axes=plt.subplots(1,3,figsize=(23,8))
for ax,k,n in zip(axes,keys,names):
 cm=np.array(rs[k]['metrics']['confusion']);im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=4)
 ax.set_xticks(range(13),labels,rotation=65,ha='right',fontsize=10);ax.set_yticks(range(13),labels,fontsize=10)
 ax.set(xlabel='预测类别',ylabel='真实类别',title=f'{n}\n完整扫描：{rs[k]["metrics"]["correct"]}/52 正确')
 for i in range(13):
  for j in range(13):
   if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>2 else '#162739',fontsize=10)
 ax.set_xticks(np.arange(-.5,13,1),minor=True);ax.set_yticks(np.arange(-.5,13,1),minor=True);ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
fig.suptitle('完整扫描混淆矩阵：每类 2 个测试 CSV，共 4 个去回程单程',fontsize=18);fig.tight_layout(rect=(0,0,1,.96));save(fig,'02_三个模型的混淆矩阵')
fig,axes=plt.subplots(3,2,figsize=(14,12))
for row,(k,n) in enumerate(zip(keys,names)):
 r=rs[k];h=r['history'];x=[v['epoch'] for v in h]
 for field,label,color,style in [('loss','训练联合损失','#587DA5','-'),('class_loss','训练分类损失','#317D45','-'),('risk_loss','训练风险损失','#BC8B3F','-'),('val_loss','验证联合损失','#CA6045','--')]:axes[row,0].plot(x,[v[field] for v in h],label=label,color=color,ls=style)
 axes[row,1].plot(x,[v['val_prefix_f1']*100 for v in h],label='验证 6 检查点平均 macro-F1',color='#2474B7')
 axes[row,1].plot(x,[v['val_final_f1']*100 for v in h],label='验证完整扫描 macro-F1',color='#D26927',alpha=.8)
 for ax in axes[row]:ax.axvline(r['best_epoch'],c='#444',ls=':',label=f'保存第 {r["best_epoch"]} 轮');ax.set_xlabel('训练轮次');ax.grid(alpha=.15);ax.legend(fontsize=9)
 axes[row,0].set_title(n+'：损失');axes[row,0].set_ylabel('损失（风险损失含动态类别权重）');axes[row,1].set_title(n+'：选模指标');axes[row,1].set_ylabel('macro-F1（%）');axes[row,1].set_ylim(0,100)
fig.suptitle('训练曲线：最终文件保存的是验证指标最优轮，而非最后一轮',fontsize=17);fig.tight_layout();save(fig,'03_训练与验证曲线')
# Derived table exports, one row per model/trial/checkpoint.
with (OUT/'逐单程逐检查点结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['模型','原CSV','采集组','方向','真实类别','扫描比例','裁剪段内秒数','预测类别','正确','最大softmax','风险分数','获认证接受'])
 for k,n in zip(keys,names):
  for row in rs[k]['test_predictions']:
   for cp in row['checkpoints']:w.writerow([n,row['path'],row['group'],'去程' if row['direction']=='out' else '回程',row['label'],cp['checkpoint'],cp['elapsed'],labels[cp['predicted']],cp['predicted']==row['true'],max(cp['scores']),cp['risk'],row['outcome']['accepted']])
pct=lambda x:f'{100*x:.2f}%'
f4=lambda x:f'{x:.4f}'
lines=[]
def add(*x):lines.extend(x)
def table(head,rows):
 add('| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |')
 for row in rows:add('| '+' | '.join(str(x) for x in row)+' |')
 add('')
add('# 十三类布料 CERA-Net 训练成果分析','',f'分析对象：`{meta["id"]}`；任务名：{meta["name"]}。',f'正式训练记录时间：{meta["created_at"]} 至 {meta["completed_at"]}。分析日期：2026-11-15。','',
'本报告分析已完成的正式训练，排除了 outputs/verification 中的程序验证模型。三份模型均为完整 CERA-Net，区别是输入通道；并非三种不同网络架构。没有重训、调参或改写原模型。','',
'## 1. 结论','',
'本次结果支持优先继续研究 ΔBx＋ΔBy 双通道：完整扫描准确率 65.38%（34/52），macro-F1 0.6275。ΔBy 和 XY 模长完整扫描均仅正确 3/52（5.77%），当前不适合作为实际分类模型。它们在中途检查点有一定识别能力，但最终状态丢失了相当一部分正确判断。','',
'双通道也尚未达到稳定部署标准：18/52 个单程分错，织棉提花缎 0/4，大衣呢和针织复合面料各 1/4；且所有模型的停止策略都因校准数据不足而未通过认证。双通道是当前最好的实验候选，不能据此宣称已实现可靠在线分类。','',
'## 2. 实验规模和可信度核查','',
'共 13 类，每类 10 CSV；每 CSV 分为去程、回程两个样本，共 260 单程。去程和回程沿用同一个采集组，划分时始终同组；260 单程不等于 260 次独立采集。')
table(['用途','CSV / 独立采集组','单程','每类 CSV'],[[{'train':'训练','val':'验证选模','cal_set':'预测集合校准','cal_risk':'停止策略风险审查','test':'最终测试'}[part],v['csv'],v['trials'],list(v['labels'].values())[0]] for part,v in audit['partition_audit'].items()])
add('“每类十组数据”实际只有每类 5 CSV、10 个单程参与梯度训练；其余用于选模、两种校准和测试。训练集 65 个采集组；测试集 26 个采集组、52 单程。','',
'复核：130 份 CSV 与 130 份配套元数据全部匹配训练清单 SHA256；源 CSV 内容无重复；各分区采集组没有交叉；各模型测试 ID 相同。重新加载冻结权重并采用流式推理，全部测试概率与保存结果最大差值为 0，混淆矩阵全部一致；验证集平均前缀 F1 也复现了保存轮的记录。该检查排除了本次发现的统计不一致或重放差异，不等于证明所有预处理物理假设正确。','',
'固定输入协议：333 Hz、名义速度 6 mm/s；去程裁剪 45–142 mm、5316 点、约 15.96 s，回程裁剪 16–114 mm、5372 点、约 16.13 s。位置窗配合固定时长扣除 0.2 s 尾部余量，并非逐条自动识别真实接触边界。ΔBy/双通道去程保留原符号、回程逐轴取绝对值；XY 模长计算 √(ΔBx²＋ΔBy²)，两方向均非负；回程未按时间反转。每通道只用训练集拟合全局均值和标准差。','',
'## 3. 总体成绩','')
table(['输入','实际/最佳轮次','训练集完整准确率','验证集完整准确率','测试集完整准确率','测试 macro-F1','去程 / 回程正确'],[[n,f'{len(rs[k]["history"])} / {rs[k]["best_epoch"]}',pct(audit['models'][k]['partitions']['train']['metrics'][-1]['accuracy']),pct(audit['models'][k]['partitions']['val']['metrics'][-1]['accuracy']),f'{pct(rs[k]["metrics"]["accuracy"])} ({rs[k]["metrics"]["correct"]}/52)',f4(rs[k]['metrics']['macro_f1']),f'{rs[k]["direction_metrics"]["out"]["correct"]}/26；{rs[k]["direction_metrics"]["back"]["correct"]}/26'] for k,n in zip(keys,names)])
add('准确率表示正确单程占比；macro-F1 对 13 类的 F1 等权平均，同时考虑该类误报与漏报。13 类均匀随机猜测的理论期望准确率为 1/13≈7.69%，这里只作参照，不是实际运行的随机分类器结果。测试规模较小，两个单通道低于该期望不代表统计上已证明劣于随机。','',
'![总体表现与扫描进度](01_总体表现与扫描进度.png)','',
'三个模型使用相同划分、随机种子 42、AdamW、学习率 0.001、batch=32、最大 100 轮、耐心 15 轮。参数量分别为 '+ '、'.join(f'{n} {audit["models"][k]["parameters"]:,}' for k,n in zip(keys,names))+'；双通道仅多 384 个参数，优势不来自明显更大的模型。但仍只有一个划分、一个种子的结果，不能作为输入优劣的普遍定论。','',
'## 4. ΔBy 单通道：当前主实验未取得合格的完整扫描分类结果','',
'训练至第 51 轮，由早停结束，保存第 36 轮；保存轮验证六检查点平均 macro-F1=0.2290，完整扫描 macro-F1=0.1603。训练分类损失从首轮 2.7497 降到保存轮 1.9358，但冻结模型在训练集完整扫描上也只有 17/130=13.08%。因此现象不只是过拟合导致测试下降：模型对已见训练样本的终点判别本身也较弱。可能涉及输入可分性、训练目标和末段状态；目前证据不能单独确定哪一项是根因。','',
'测试完整扫描只识别对亮丝粗花呢 2/4、平纹棉布 1/4，其余 11 类全部 0/4。最终 19/52 被预测为织棉提花缎、10/52 被预测为平纹棉布，合计 55.77%；真正的织棉提花缎却 0/4，存在明显的终点类别偏向。加厚1x1罗纹布、毛织物面料、覆膜帆布三类各 4/4 都被分成织棉提花缎。','',
'单通道并非全程没有学到信息：50% 检查点正确 20/52=38.46%，到完整扫描只剩 3/52。75%→100% 期间，有 16 个原本正确的单程被改错，仅 2 个原先错误的变正确。这说明“收集更多点后一定更准”的假设不符合本次结果。','',
'## 5. ΔBx＋ΔBy：当前最有价值的实验模型','',
'训练达到 100 轮上限，保存第 95 轮；保存轮验证六检查点平均 macro-F1=0.7317，完整扫描 macro-F1=0.7744。训练分类损失由 2.8065 降到 0.6395。冻结模型训练集完整扫描 95/130=73.08%，验证集 21/26=80.77%，测试集 34/52=65.38%。验证集较小且用于选模，它高于训练集不奇怪；仅凭这些数值不能判断没有过拟合，也不能把验证分数当作部署精度。','',
'完整扫描亮丝粗花呢、平纹棉布、毛织物面料、灯芯绒四类均 4/4。亮丝粗花呢 F1=1.000；平纹棉布 F1=0.800，说明虽然没有漏掉其真样本，但有其他布料被误报成平纹棉布。弱项是织棉提花缎 0/4，大衣呢、针织复合面料各 1/4。织棉提花缎的 4 段分别被分为平纹棉布 2 段、亮丝欧根纱 1 段、覆膜帆布 1 段。','',
'方向方面：去程 14/26=53.85%，回程 20/26=76.92%，相差 6 个单程。回程取绝对值后仍保留分类信息，没有证据支持丢掉回程。但两方向尚未达到相同表现，不能认为绝对值已经消除了所有方向相关差异。26 个测试 CSV 中，12 个去回程都对，10 个只对一个方向，4 个两个方向都错；这不是合并去回程的分类成绩。','',
'20% 检查点正确 41/52=78.85%，macro-F1=0.7812，约裁剪段开始后 3.2 s（不含之前悬空/接近平台的时间）；50% 为 38/52=73.08%。但测试集上发现 20% 最好属于事后观察，不能直接据此把在线停止点改为 20%。冻结模型在验证集的最高准确率反而出现在 50%=22/26=84.62%，下一轮必须由验证数据决定方案并用新的独立采集验证。','',
'## 6. XY 模长：丢失轴向信息后没有带来收益','',
'训练至第 31 轮早停，保存第 16 轮；保存轮验证六检查点平均 macro-F1=0.1947，完整扫描 macro-F1=0.0692。训练分类损失由 2.7495 降至保存轮 2.0551。冻结模型训练集完整扫描 14/130=10.77%，测试集 3/52=5.77%，同样不只是测试泛化问题。','',
'完整扫描只对亮丝粗花呢、平纹棉布、空气层各 1 段，其余 10 类全部 0/4。最终 21/52 被分成织棉提花缎，9/52 被分成平纹棉布；加厚1x1罗纹布、毛织物面料、覆膜帆布各 4/4 被分成织棉提花缎。30% 时达到本模型测试检查点最高的 19/52=36.54%，最终显著退化。','',
'检查原始预处理输入，ΔBy 主导磁场变化幅值。逐单程计算 XY 模长与 |ΔBy| 的 Pearson 相关系数，再对 130 个去程/回程分别取中位数，得到去程 0.99973、回程 0.99395。ΔBx RMS 占 XY 总 RMS 的中位比例，去程约 3.63%、回程约 18.39%。因此模长在数值上接近 |ΔBy|；较小的 ΔBx 在平方求和中被压弱，同时两个分量的组合关系被压成一个数。这个结果与单通道性能相近、双通道性能更好相符；它支持保留两轴分别输入，但尚不能从此确定只有 ΔBx 一轴就足够（本次没有训练 ΔBx-only 对照）。','',
'## 7. 每一类布料的完整扫描成绩','',
'每格是正确单程数 / 4，括号内为召回率。每类实际仅 2 个测试 CSV；一个单程变化就使召回率变化 25 个百分点。')
table(['布料','ΔBy','ΔBx＋ΔBy','XY 模长'],[[lab]+[f'{round(rs[k]["metrics"]["per_class"][i]["recall"]*4)}/4 ({pct(rs[k]["metrics"]["per_class"][i]["recall"])})' for k in keys] for i,lab in enumerate(labels)])
add('![三个模型完整扫描混淆矩阵](02_三个模型的混淆矩阵.png)','',
'## 8. 扫描进度与终点退化','')
table(['已扫描比例','ΔBy 正确/52','双通道 正确/52','XY 模长 正确/52'],[[f'{q} %']+[f'{rs[k]["prefix_metrics"][i]["correct"]}/52 ({pct(rs[k]["prefix_metrics"][i]["accuracy"])})' for k in keys] for i,q in enumerate(qs)])
add('六检查点 macro-F1 的测试平均值分别为 ΔBy 0.2007、双通道 0.6678、XY 模长 0.1772；这是一个独立于最终准确率的“整个扫描过程表现”摘要，不应与最终 F1 混用。','',
'检查 260 个预处理单程的末 10% 与中间 45–55% 窗口：逐单程 ΔBy RMS 比值的中位数，去程 1.027、回程 1.010，末段并未整体衰减到悬空零基线。因此现在不能简单断言“末段都已离开布料”，也不能仅靠幅值排除边缘过渡、姿态变化或裁剪边界问题；需要与真实接触位置核对。','',
'模型的最终分类直接读取最后时刻的证据状态；此前预测不会被自动多数投票保留。证据递推含可学习遗忘系数，GRU 也不保证长期保存所有早期信息。因此框架不强制准确率随扫描时间单调上升。当前已确认的是训练/测试中均存在终点弱化，物理接触窗、末段信息和状态保持的相对贡献仍待单独实验确认。','',
'## 9. 校准与风险头：全部尚不能认证提前停止','',
'三模型均为 `insufficient_evidence`，停止阈值 theta=null，六个预测集合分位数全部为 null，测试认证接受率 0%、平均扫描比例 100%。这表示没有通过当前认证条件，不能把它解释为分类器完全没有输出，也不能把接受后错误率 N/A 解读为 0%。当前回退到完整扫描输出，但该输出仍未获“错误率≤5%”认证。','',
'预测集合校准只有 13 个独立采集组。95% 分位数的有限样本秩为 ceil((13+1)×0.95)=14，超过可用 13 组，程序按无穷分位数处理，集合包含全部 13 类，无法成为单一类别。至少 19 个独立组才可能得到有限的 95% 分位数；这仅解除分位数障碍，不保证集合足够小或满足后面的逐类风险审查。','',
'风险审查每类仅 1 个独立组，还同时审查 63 个候选策略；当前逐类上界约 2.2027，无法满足通过条件。这里输出的是审查损失的上界，不是“错误概率 220%”。因此只提高训练轮数或提高分类准确率，也无法解决当前的认证样本量障碍。','')
table(['模型','风险 Brier（越低越好）','六检查点平均风险分数','六检查点实际错误比例'],[[n,f4(rs[k]['brier']),pct(np.mean([cp['risk'] for row in rs[k]['test_predictions'] for cp in row['checkpoints']])),pct(np.mean([cp['predicted']!=row['true'] for row in rs[k]['test_predictions'] for cp in row['checkpoints']]))] for k,n in zip(keys,names)])
add('风险头使用动态加权二分类损失；原始分数并未校准成个体错误概率。上表只作分数与实际错误的诊断，不能直接把 0.4 分数当作 40% 的个体出错概率。每模型 52×6=312 个风险观测来自同一批 52 单程、26 个采集组，彼此相关，不能当作 312 次独立实验。Brier 反映风险分数误差，不是分类准确率，也不应单独决定哪个模型可部署。','',
'## 10. 训练曲线如何解读','',
'![训练与验证曲线](03_训练与验证曲线.png)','',
'选模优先看验证集六检查点平均 macro-F1，相同时再比较完整扫描 F1、验证联合损失；因此保存的最佳轮次不是最低损失轮，也不必是最后一轮。双通道第 100 轮验证完整 F1=0.5158，明显低于保存的第 95 轮 0.7744；实际模型文件保留第 95 轮。','',
'训练联合损失=分类交叉熵＋加权风险 BCE。风险权重随每轮训练错误分布更新，且训练时开启 dropout、验证时关闭，因此联合损失的跨轮变化与训练/验证差值需要谨慎解释。本次没有保存逐轮训练准确率；本报告的训练准确率来自对最佳冻结模型的补充评估，不能用来代替每轮学习曲线。','',
'## 11. 基于此次结果的下一步','',
'1. 保留 ΔBy 为主实验结果如实报告，但以双通道作为下一步改进基线。当前证据不支持把 XY 模长作为首选。','2. 优先定位终点退化：核对裁剪窗口与真实接触位置，比较验证集上预先选定的中途检查点与完整段汇总方案。修改后使用新的采集组做最终测试，避免继续用当前测试集选方案。','3. 对织棉提花缎、大衣呢、针织复合面料检查重复性和标注/采集条件，并增加独立采集；不要把同一 CSV 切出的更多窗口计作新增独立实验。','4. ΔBx-only 对照、独立方向评估和多种子分组验证可分别检验“额外信息来自哪里”“回程处理是否合适”“单次结果是否稳定”；这些本次尚未完成，报告不将其当成结论。','5. 把分类准确率提升与停止认证数据量分开处理。当前逐类小样本严格风险界很保守，需要重新核算目标与独立校准采集量，不能通过调低风险分数制造认证通过。','',
'## 12. 产物与可重复性','',
'- `冻结模型复核.json`：源完整性、分区核查、冻结模型五个分区的六检查点指标及输入统计。','- `逐单程逐检查点结果.csv`：3×52×6=936 行，可按模型、布料、方向逐条检查。','- 三张图同时提供 PNG 和矢量 PDF。','- `复核模型.py` 与 `生成分析报告.py`：只读原始数据/模型，写派生结果，无训练调用。','',
'原始证据：`'+str(RUN/'metadata.json')+'`，以及三个子目录中的 `model.pt`、`history.json`、`result.json`、`calibration.json`。','')
(OUT/'训练成果详细分析.md').write_text('\n'.join(lines),encoding='utf-8')
print('Report and figures written:',OUT)
