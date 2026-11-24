# 采集端在线转发安装与预览记录

用户确认本批采集已结束，并明确允许修改原采集程序后，已把经过离线验证的实时转发功能安装到 `xz-control/`。**功能在下次人工重启原采集程序后可见，默认关闭。** 未重启/停止任何现有进程，没有打开串口或执行硬件/电机命令；采集数据 CSV 未改动。没有提交 Git。

原应用仅修改 `xz_workbench/backend.py`、`xz_workbench/ui.py`，新增 `xz_workbench/realtime.py` 和两个离线测试文件。原窗口标题、设置目录、配置、默认输出目录保持原样；未用预览副本的设置隔离逻辑覆盖原应用。

安装前已检查原文件与预览的差异仅为本次转发功能（以及预览专用标题），并在写入前再次核验备份 SHA256。原两个源文件的安装前副本、原/安装后 SHA256，以及新增文件原本不存在的记录保存在 `布料分类训练脚本/outputs/acquisition-online-preview/original-backup/manifest.json`；只备份源码，不复制数据或虚拟环境。

## 人工使用

下次人工重启原采集程序后，UI 需勾选「允许实时转发（仅本机）」并点击「开始转发」；「停止转发」和取消勾选均关闭。不会随设置保存自动开启。仅开启时监听 `127.0.0.1:8766`。不要同时运行预览副本和原应用连接同一串口。

所有 socket 收发由独立广播线程承担，采集帧入口仅做字段处理和有界非阻塞入队。默认队列 2048 项；溢出丢旧并发送 reset；客户端积压超过 64 KiB 或 0.5 秒则断开。最多四个客户端。绑定失败、网络异常或客户端断开只影响转发状态，不中断记录。

双方已协调 TCP UTF-8 NDJSON 协议：sample 包含 capture_id、stage、t_s、x_mm、by_uT、baseline_by_uT、valid、invalid_reason、motor_fault；motor_fault_code 为可选诊断字段。By 是原始 By，基线来自实际 baseline[1]；X 来自控制器原始脉冲 / 当前 x_ppmm；t_s 来自解绕传感器微秒 / 1e6。在线端重新执行 By 专用清洗，不使用 CSV 全局 XYZ 质量位。

仅 step_003 / step_004、录制中且基线已知、非校准/重启、控制器正常的样本可分类。其余样本标 invalid，消费者清空窗口。会话/阶段/设备时钟段/基线/有效录制状态变化发送 reset，会话结束发送 end。首次连接发送 client_connected reset；转发停止会关闭连接，消费者也须在 EOF / 重连时清空状态。

## 保留的独立预览

`布料分类训练脚本/outputs/acquisition-online-preview/` 仍是独立源码副本，未启动。除后来增加的原源文件备份外，它不包含原始采集数据、Git、虚拟环境或已有二进制；预览自身设置与默认输出位于副本目录。人工演练入口为 `launch_preview.py`；`--help` 不导入 Qt 或设备；`--demo` 仅模拟。

最初开发阶段未编辑原 `xz-control/`，待用户明确确认采集完成后才安装上述三个应用源文件。预览 README 的隔离设置说明仅适用于副本，原应用仍使用原设置与输出路径。

## 验证

采用 TDD：新增模块缺失、后端转发缺失、UI 控件缺失及空闲客户端忙循环分别先失败，再实现/修复。测试覆盖 loopback NDJSON、未启用不监听、绑定失败、客户端退出、慢客户端丢弃、队列溢出 reset、By/基线/坐标/时间单位、设备时间解绕、阶段/时钟/基线 reset、end、转发异常隔离，以及 Qt 复选框与开始/停止。

预览最终完整测试：205 tests / 26.322 s / OK。原目录安装后 focused 测试：11 tests / 0.283 s / OK。使用原虚拟环境解释器，命令加 `-B`；原测试设备端均为模拟器、注入字节流或 fake UI boundary，未打开实物串口。已有固件测试只在临时目录编译、运行模拟场景，不刷写硬件。

原 preferences.py 与安装前预览复制版本字节一致；catalog.py 与预览的差异仅为预览原有的设置隔离路径，验证原设置默认值未替换。全部备份与安装文件哈希已写入 manifest。

原目录最终完整测试：在 `/home/serein/classify-cloth/xz-control` 执行 `QT_QPA_PLATFORM=offscreen .venv/bin/python -B -m unittest discover -s tests`，**186 tests / 25.278 s / OK**。仅有 Qt offscreen 的预期布局/raise 提示。
