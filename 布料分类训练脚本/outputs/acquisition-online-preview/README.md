# 独立采集转发预览（尚未启用）

此目录是 `xz-control` 必要应用源码的独立副本，不含原始数据、设置、虚拟环境、Git 历史或二进制文件。开发此副本时未修改或启动原应用。此后用户确认采集完成并明确授权，已将转发模块及后端/UI开关安装到原应用；原设置和数据路径保持原样，功能在人工重启后可见且默认关闭。运行中的进程未被重启、停止或连接。此副本仍未启动。

实物运行前，先关闭原采集程序，再启动此副本，避免两个程序争用串口。不要在当前批次进行中启动此预览。脚本不自动连接实物、不自动使能或运动。

在带有 `requirements.txt` 中依赖的 Python 环境中执行：

```bash
python /home/serein/classify-cloth/布料分类训练脚本/outputs/acquisition-online-preview/launch_preview.py
```

仅模拟演练可加 `--demo`。预览使用本目录 `preview-runtime/settings.json` 独立保存设置，默认采集输出位于本目录 `data/experiments_linux`，不读取原应用保存的配置。操作者仍可在 UI 选择保存目录，应选择新的预览目录。

界面新增「允许实时转发（仅本机）」复选框和「开始转发 / 停止转发」按钮。默认关闭，不随设置持久化。先勾选，再明确点击开始；取消勾选会停止转发。仅开启转发时监听 `127.0.0.1:8766`，最多四个本机客户端。监听失败仅显示转发失败，不影响采集。

TCP 服务向每个连接发送 UTF-8 NDJSON（每行一个 JSON），客户端只需读取。首次连接收到 reset。接收原始帧后转发实际 By、实际基线 By、控制器 X 脉冲 / 当前 x_ppmm，以及解绕传感器微秒 / 1e6；不使用主机时间替代传感器时间，不使用基于 XYZ 的全局有效标记。

```json
{"type":"sample","capture_id":"session-id","stage":"step_003","t_s":12.004,"x_mm":5.0,"by_uT":37.25,"baseline_by_uT":30.0,"valid":true,"invalid_reason":"","motor_fault":false,"motor_fault_code":"NONE"}
```

仅 `step_003`、`step_004` 且有采集 session、记录器、已知基线、正常控制器状态时标为 valid。校准、采集板重启、其他阶段、未录制、故障、缺少基线或坐标时标为 invalid；它们可作诊断，消费者必须清空窗口，不得分类。`by_uT` 为原始 By，在线消费者自行执行 By 清洗/减基线/平滑，不沿用 CSV 的全局质量标记。未连接控制器时 `motor_fault=true`、code 为 null，坐标为 null，样本无效。

会话、阶段、设备时钟段、基线、录制有效状态发生变化时发送 `{"type":"reset","capture_id":"...","stage":"...","reason":"context_changed"}`。会话结束发送 `{"type":"end","capture_id":"...","stage":"...","reason":"completed"}`；消费者也必须在 EOF / 重连 / 无效样本时清空状态。

采集线程只执行有界非阻塞入队；独立广播线程负责所有 socket 收发。队列默认 2048 项，溢出丢弃旧项并插入 reset，UI 显示累计丢弃数。客户端待发超过 64 KiB 或积压超过 0.5 秒则关闭连接，客户端需重连。转发异常不会进入录制或电机故障处理流程。网络转发属于尽力发送，不作为采集数据的可靠存储。

离线测试：`QT_QPA_PLATFORM=offscreen python -B -m unittest discover -s tests`。这些测试仅使用临时文件、模拟器和 loopback socket，不访问实物设备。`launch_preview.py --help` 不导入 Qt，也不打开设备。
