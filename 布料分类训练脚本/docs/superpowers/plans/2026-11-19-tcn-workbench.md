# TCN 专用工作台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax.

**Goal:** 交付只显示TCN/|dBy|、可调结构与900点/四组研究的桌面训练工作台。

**Architecture:** 保留旧ModernClassifier与旧配置序列化，显式tcn_spec启用新模型和父单程窗口训练。所有任务冻结candidate结构；验证/测试/识别复用窗口聚合。原桌面作为兼容基础类，新TCN工作台默认启动，复用数据审查和成果流程。

**Tech Stack:** Python3.10、PyQt5、PyTorch、NumPy、pytest。

**Spec:** ../specs/2026-11-18-tcn-focused-ui-design.md 和 ../specs/2026-11-18-tcn-four-way-design.md（用户已批准开始实施）。

## Global Constraints

- 只对新TCN专用任务启用新结构；输入|dBy|、333Hz、abs-v3。
- 同源采集/会话先分区后切片；尾窗保留；scaler仅训练父单程拟合。
- batch_size单位为父单程；全部窗反向累积后一步更新；验证按有效长度加权概率平均。
- 单配置五折，四组20任务；默认四组结构锁定，不测试选型。
- 旧配置、旧bundle及恢复binding保持原结构；不改原始CSV、已有实验；不启动正式全量训练。
- 在现有feature分支执行（不在main/master），保留用户未跟踪.superpowers文件。

## Review Focus

- 旧配置新增默认字段导致旧恢复失败：Task1序列化回归与Task2旧恢复测试。
- 尾窗、不同父窗数和尾批权重错误：Task2微批等价测试。
- 保存宽度与实际构建不一致：Task3候选篡改/最终加载测试。
- 隐藏GUI选项仍提交其他模型：Task4请求和旧项目限制测试。
- 小屏和高DPI挤压控件：Task4多尺寸截图和几何测试。

### Task 1: TCN结构和候选规范

**Files:** Create cloth_trainer/tcn_models.py, cloth_trainer/tcn_study.py, tests/test_tcn_study.py; Modify cloth_trainer/modern_config.py.
**Interfaces:** TCNClassifier(channels,classes,spec); normalize_spec(spec), default_spec(); candidates(cfg)->list[dict]; window_slices(length,size), split_batch(x,lengths,size)->(windows,lengths,parents,weights,starts); aggregate_log_probs(logits,parents,weights,count).
- [x] 写结构/因果/mask、精确尾窗、极端概率、配置兼容与四候选失败测试。
- [x] Run `.venv/bin/python -m pytest -q tests/test_tcn_study.py`，确认缺少模块失败。
- [x] 最小实现，测试通过；提交本任务代码与测试。

### Task 2: 父单程窗口训练

**Files:** Modify cloth_trainer/modern_training.py; Create tests/test_tcn_training.py.
**Interfaces:** fit/evaluate新增可选candidate参数；新模型保存_model candidate；clip(model,max_norm=1)；窗口父批共享loader和旧恢复机制。
- [x] 写非等长父批/尾批微批等价、更新数、聚合父行、恢复及clip阈值失败测试。
- [x] Run `.venv/bin/python -m pytest -q tests/test_tcn_training.py`，确认失败。
- [x] 实现窗口loss、验证聚合NLL、记录updates/points/memory、可调clip/scheduler和恢复绑定。
- [x] Run `.venv/bin/python -m pytest -q tests/test_tcn_training.py tests/test_modern_training.py`；提交。

### Task 3: 研究编排与成果生命周期

**Files:** Modify cloth_trainer/modern_engine.py, engine.py, catalog.py; Create tests/test_tcn_engine.py.
**Interfaces:** 显式study含revision/candidates；configuration_id含候选；final bundle包含冻结candidate/model_revision；加载严格验证并构建TCNClassifier；旧无study任务走原路径。
- [x] 写单配置与20任务CV->final->test->export/import->predict测试及篡改候选拒绝测试。
- [x] Run `.venv/bin/python -m pytest -q tests/test_tcn_engine.py`，确认失败。
- [x] 实现共享冻结候选、编排、独立汇总与paired fold差值、最终候选加载。
- [x] Run `.venv/bin/python -m pytest -q tests/test_tcn_engine.py tests/test_modern_engine.py tests/test_recognition.py`；提交。

### Task 4: TCN专用GUI与布局

**Files:** Create cloth_trainer/tcn_ui.py, cloth_trainer/tcn_settings.py, tests/test_tcn_ui.py; Modify ui.py（仅可复用hook）, theme.py, train_cloth.py, plots.py, README.md.
**Interfaces:** TCNTrainerWindow复用TrainerWindow后台，3页导航，controls_config强制TCN/By；TCNSettingsDialog编辑结构/模式/窗口/训练参数；旧TrainerWindow测试兼容。
- [x] 写专用请求/5与20任务/设置生效/小屏无裁切/旧项目只读测试，运行确认失败。
- [x] 实现滚动参数与监控tabs、可调结构、冻结四组、结果筛选；默认入口使用新窗口。
- [x] Run `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_tcn_ui.py tests/test_ui.py tests/test_startup.py`。
- [x] 1280×720、1366×768、1920×1080及缩放截图检查；合成小流程与实际CUDA前反向；完整回归。
- [x] 更新README、提交；fresh reviewer检查全改动，处理重要问题后复测，记录交付状态。

完成记录：见 `../tcn-workbench-acceptance-2026-11-19.md`。最终完整回归220项通过；独立审查反馈已处理。
