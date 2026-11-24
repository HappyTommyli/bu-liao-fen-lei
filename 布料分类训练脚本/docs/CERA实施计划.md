# CERA-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 原始CSV导入后，自动裁剪并因果预处理，端到端训练CERA-Net、独立校准、保存成果及离线流式识别。
**Architecture:** config/data定义协议和单程；model定义并行因果网络；calibration实现冻结后的组级接受审查；engine串联训练和成果；独立PyQt5五页桌面通过子进程调用同一核心。
**Tech Stack:** Python3.10+, NumPy, SciPy, CPU PyTorch, PyQt5, Matplotlib。
**Spec:** docs/CERA-Net最终训练方案.md（用户已授权执行）。

## Global Constraints
- 全部代码在本文件夹，原始CSV只读；已删除错误代码和其独立Git历史，不留档。
- ΔBy主实验；双通道、XY模长对照；同CSV去回程同组。
- 333Hz，去程45–142mm，回程16–114mm；因果保持、20ms缺口上限。
- TCN32/64/64，kernel5,dilation1/2/4；GRU64；门控64、证据64、风险67→64→32→1。
- 六个固定前缀检查点；CE+WBCE联合训练；冻结后预测集合和组风险审查。
- 每类10独立组最低，5/1/1/1/2分区；分类可训练但风险认证可因样本不足失败。

## Review Focus
- 时钟重置／无效点：不跨长缺口、不用未来插值；固定时长不足须拒绝。
- 合并会话或重复CSV：来源组不跨分区，组风险权重一致。
- 隐含未来信息／状态丢失：整段与逐块输出一致，未来替换不影响前缀。
- 校准不足／无接受：明确不足、N/A，不能把风险分数当认证。
- 子进程取消／异常：非成功状态、日志保留，已校准权重不可重训后复用。

### Task 1: 因果数据和分区
Files: cloth_trainer/config.py, data.py; tests/test_data.py。
Interfaces: DataConfig, load_record(path, config, label=None), import_records(paths,config,overrides), prepare_trials(records,signal), split_records(records,seed)。
- [x] 失败测试：裁剪/方向处理、未来替换、长缺口、重复、四阶段分区和合并组。
- [x] 运行pytest tests/test_data.py，预期缺模块失败。
- [x] 实现；测试通过；扫描实际目录，不修改CSV。

### Task 2: 因果网络和校准
Files: model.py, calibration.py; tests/test_model.py, test_calibration.py。
Interfaces: CERANet(channels,classes,variant), forward(x,state=None), forward_stream(x,state), joint_loss(outputs,targets,indices,weights), calibrate(set_rows,risk_rows,labels,family_size), stop_trial(rows,state)。
- [x] 失败测试：多前缀因果、逐块等价、两损失共享梯度、独立组分位数、空接受与保守上界。
- [x] 实现显式状态和有限候选审查；运行上述测试通过。

### Task 3: 联合训练、成果与识别
Files: engine.py, storage.py, train_cloth.py; tests/test_engine.py。
Interfaces: train_request(request,emit,cancelled), predict_request(request,emit,cancelled), bundle加载；统一JSON任务。
- [x] 失败测试：真实小网络训练、保存重载、三输入共享分区、权重冻结、源变更、取消、成果包。
- [x] 实现训练/验证/集合校准/风险校准/测试，流式回放与报告；运行核心完整测试。

### Task 4: 五页UI和启动
Files: ui.py, plots.py, theme.py, requirements.txt, 安装依赖.sh, 启动布料分类训练.sh, README.md; tests/test_ui.py。
- [x] 失败测试：目录导入、标签选择、训练快照、项目恢复、历史和识别、子进程退出。
- [x] 实现简洁五页，后台导入、训练/取消、结果和历史、预测、导入导出。
- [x] 实际CSV短训练与识别、全套pytest、各页截图检查、异目录启动。
- [x] 独立代码审阅，修复重要问题，最后提交。
