# 工作台：布料命名、采集板动作与默认方向键

用户已在对话批准实施。范围：现有 Linux 工作台；不改既有数据、Windows CSV 列、固件或电机协议。

## Task 1：命名与数据管理
- 独立目录/CSV 同名：布料_001/布料_001.csv；每个名称独立递增，保留用过的编号。
- metadata 记录 label/run_name/sequence；记住上次成功采集或重命名的标签。
- 已结束的 completed/aborted 记录可重命名；活动记录、后台写盘、interrupted 记录禁止重命名。
- 兼容旧 sensor.csv；重命名不改变 CSV 字节。
- RED→GREEN：编号冲突、持久化、旧记录迁移、重命名阻止和失败回滚。

## Task 2：非阻塞动作与分段采集
- 表格新增 restart_sensor（默认超时10s）、baseline（默认2s）；旧流程兼容。
- REBOOT 只发采集板；需启动标记和新样本，不将旧数据误作重启成功。
- 重启使基线失效；采集流程重启后必须基线，录制分段且每份只有一个基线。
- 首步重启不生成空文件；超时/中止取消后续动作、待采集状态，电机心跳持续。
- RED→GREEN：外部动作等待、预检、重启成功/无响应/断连、分段与取消。

## Task 3：界面
- 主采集区域增加名称与下一次文件名；数据管理放采集设置。
- 数字和图只显示ΔB，无基线显示待校准，新基线清历史。
- 删除键盘开关；应用内方向键始终用于移动，输入框和表格不响应方向键编辑；失去焦点停止。
- RED→GREEN：键盘焦点、默认可用、重复事件、动作往返、差分显示。

## Task 4：交付验证
- 全套 unittest、离屏界面截图、启动入口检查；独立最终代码审查。
- 比较原目录初始哈希，仅复制本次拥有的源文件、测试、文档，原数据保持字节不变。

## Ledger
- Baseline：88 tests passed（2026-11-13）。工作区为已有原生隔离 worktree 下源代码快照；原仓库存在用户改动。
- 接口：Catalog reservation→Recorder named output→Worker sessions→UI；PlannedStep.kind→Engine external callback→Worker nonblocking sensor state。两处接口一起测试。
- 执行使用 executing-plans、test-driven-development 和 verification-before-completion；不自动提交原仓库，不打开真实串口。

- Task 1 complete：Catalog 编号/迁移/阻止活动记录/失败回滚；Recorder 保持 23 列。
- Task 2 complete：外部动作严格等待完成；实际格式串口替身覆盖 REBOOT、READY、断连、旧帧、超时和电机 PING。
- Task 3 complete：深色界面、命名预览、管理入口、默认方向键、Delta 显示；输入框/面板/下拉表格测试通过。
- Validation：114 tests passed，包含原 88 项。新功能测试先失败后实现；补充了磁数据显示陈旧快照的回归。
- Task 4 complete：独立审查问题已修复；已交付原目录，并在原目录再次通过 118 项测试、启动器 --help、bash 语法和编译检查。

- Final review：独立只读审查，无 Critical；两项 Important：分段保存失败未取消当前流程、重复运行请求误中止现有采集。
- Final: fixed segment failure — test_delayed_segment_save_failure_aborts_its_remaining_recipe RED→GREEN；test_delayed_old_session_failure_does_not_abort_new_capture 保护跨会话隔离。
- Final: fixed duplicate request — test_duplicate_run_record_request_does_not_abort_first_capture RED→GREEN。
- Final: Ruling: 普通函数被空布料名称阻止按 Important 处理 — 用户只运行电机函数时不应依赖采集名称 — 若判断不当的代价为普通运行允许空名称，但普通运行不创建 CSV。test_ordinary_run_does_not_require_a_fabric_label RED→GREEN。
- Final verification：118/118 unittest passed，启动器 bash 语法、--help、Python 编译和实际字体的离屏界面渲染通过。
- Deferred minors：无。无未解决的重要问题；实物串口和已部署固件的验收需用户接硬件进行。

## 2026-11-13 实机重启超时排查
- 用户反馈：原提示“采集板重启超时：未确认启动和新数据，请核对固件 REBOOT 支持”。
- 已核查：发送 REBOOT 不依赖板型识别；USB VID/PID/序列号用于定位设备和重连。系统可枚举两个 Espressif USB 设备，运行中的工作台持有两个串口。
- FYP fixed90 源码实现 REBOOT→确认→ESP.restart→READY，相关 11 项固件/旧采集器测试通过；当前实际烧录固件版本未知。
- 已确认的脚本问题：原超时消息没有区分回应、启动、READY、有效帧和 USB 重连阶段，无法从用户给出的消息定位硬件端原因。
- 修改：阶段日志、接收字节与四列行数、最后非数据响应、重连错误；保持原成功判定和控制权不变。不进行实机串口访问、固件上传或强制重启工作台。
- Tests：9 项新增诊断测试（先观察失败）；127 项完整 unittest 通过。原数据保持不变。
- 独立审查：发现 READY 后断连的提示优先级应显示当前重连错误，新增回归 RED→GREEN 后修复；无未解决的重要代码问题。
- 实机超时根因仍需重开工作台、再执行重启，通过新分阶段日志确认；本次交付为诊断改进，不宣称已修复实际板端问题。
