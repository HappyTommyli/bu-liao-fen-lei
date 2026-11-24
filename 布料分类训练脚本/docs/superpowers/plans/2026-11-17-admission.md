# 数据审查与训练准入 Implementation Plan

> For agentic workers: use executing-plans, inline implementation and one final independent review.

Goal: 自动审查、显示所有状态、只允许有资格的采集组进入现有五折分类流程。
Architecture: quality.py负责单次判断，admission.py负责固定批次及资格校验；data.py提供可追溯观测，dataset/engine保存并执行准入，ui只消费同一结论。
Tech stack: 现有Python/NumPy/PyQt5/PyTorch，不加依赖。
Spec: ../specs/2026-11-17-admission-design.md

## Global Constraints
- 原始文件、基线、波形、模型代码、训练参数、旧实验和混淆矩阵保持原样；不启动正式训练。
- 历史7次双指纹排除，11次复核等权；四条件同时满足才新判异常；缺依据不准入。
- 全部候选批次固定，取消勾选不重算类别比例；组计数不算去回两次；≥10组且≥80%才整类停用。

## Review Focus
- 同名新采集不永久排除，CSV相同metadata不同也不能误匹配。
- 有效采样中位数与位置箱中位数不同；未标准化、无效点不参与、共同区间不越界。
- 多CSV同会话、混合标签/结论、不足10组以及增量导入不能通过选择子集绕过。
- 直接CLI、旧快照、续训、修改标签与筛查报告篡改不能绕过新实验准入。
- 旧完整实验可复现评价/识别，数据错误和未知状态仍显示且不影响合法数据训练。

## Task 1: 单次审查
Files: quality.py, data.py, resources/admission_annotations.json, tests/test_admission.py, tests/test_data.py fixture.
Interfaces: Record.audit, annotate(record), direction_metrics(record), current_rule().
- [x] RED tests: 7/11双指纹、同名新记录、四项缺一、噪声缺失、共同中位数、完整性报告。
- [x] 实现版本化五种状态与可追溯指标，保留旧资源和旧加载规则；合成测试补明确噪声依据。
- [x] GREEN focused suite and commit.

## Task 2: 固定批次与快照/训练准入
Files: admission.py, dataset.py, engine.py; tests/test_admission.py,test_dataset.py,test_engine_v3.py.
Interfaces: review_batch(records,errors,candidates,config), validate_batches, eligible_records, require_admission; snapshot.admission。
- [x] RED tests: ≥80%且10独立组、缺依据/同会话、选择不改比例、批次校验、旧快照只读/新训练拒绝。
- [x] 实现组汇总与不可变批次，完整审查报告进入快照/实验；worker从完整批次检查并仅对资格记录五折，续训检查版本。
- [x] GREEN synthetic workflow and commit.

## Task 3: 简洁界面与报告
Files: ui.py, train_cloth.py; tests/test_ui.py.
- [x] RED tests: 排除/无法判断不可勾选但显示、详情指标、汇总、增量批次固定、标签变化重新审查、CLI审查输出。
- [x] 接入同一准入结论、保留报告/错误行/类别停用、快照旧版提示；不改变识别混淆矩阵。
- [x] GREEN UI/QProcess checks and commit.

## Task 4: 验证与交付
- [x] 全套测试、只读310CSV审查，7/11与分子分母核验，原文件哈希不变；GUI截图/报告导出/旧模型识别回归。
- [x] 最终独立审查一次，重要问题RED→GREEN修复；README/实施记录、保存证据与提交；不正式训练。
