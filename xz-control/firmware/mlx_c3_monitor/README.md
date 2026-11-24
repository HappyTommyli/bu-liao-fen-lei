# XIAO ESP32-C3 采集板：REBOOT 兼容补丁

这份固件以用户提供的 `/home/serein/classify-cloth/mlx_c3_monitor.ino` 为基础。该源文件的目标是 Seeed XIAO ESP32-C3，FQBN 为 `esp32:esp32:XIAO_ESP32C3`；其中出现的 S3 指接线方案。本次通过 ROM 实测确认芯片为 ESP32-C3，Flash 为 4MB。

原程序没有读取主机串口命令，因此工作台发出 REBOOT 后会继续正常采集，最终等待超时。

## 本次修改

- 增加完整行 `REBOOT\n` / `REBOOT\r\n` 解析。大小写错误、附加参数、超长行不会触发重启。
- 先返回 `# fw=mlx_c3 reboot requested`，flush 并等待 200ms，然后调用 `ESP.restart()`。
- 启动后增加 `# fw=mlx_c3 build=reboot-v2 target=XIAO_ESP32C3` 便于识别固件。
- 正常采样、校准等待和未找到传感器的等待期间都能接收重启。
- 优先探测本机实测 SDA=3/SCL=4，保留其余原引脚候选，避免先探测未接线引脚拖延重启。串口、增益、分辨率、过采样、滤波、采样周期和四列原始数据输出保持原代码，不需要改线。

原四列仍为 `t_us,Bx_uT,By_uT,Bz_uT`，115200 波特率，采样周期 3000µs。基线和 Linux 工作台的 23 列 CSV 保存由原工作台处理。

## 使用

Arduino IDE 打开本目录的 `mlx_c3_monitor.ino`。`serial_commands.h` 必须与它放在同一目录。

选择原源码指定的 **XIAO_ESP32C3** 开发板，保持 USB CDC On Boot 启用；安装 **Adafruit MLX90393 2.0.5** 及其依赖。先编译：

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 /home/serein/classify-cloth/xz-control/firmware/mlx_c3_monitor
```

上传需另外选择实际采集板串口，并结束工作台对该串口的占用；上传对象是采集板。源码保存到电脑后，板上的旧程序不会自动改变，必须烧录新版才支持 REBOOT。

启动后应看到固件版本、实际 SDA/SCL 和 READY，再收到原四列数据。工作台重新连接传感器，执行“重启采集板 → 采集空载基线”验证；确认日志顺序为 REBOOT 确认、启动、READY、新数据。

## 本次验证

使用 Arduino CLI 1.6.0、Espressif Arduino core 2.0.17、Adafruit MLX90393 2.0.5，按 `esp32:esp32:XIAO_ESP32C3` 实际编译通过。程序占 271574 字节，静态内存占 14060 字节。`compiled-c3` 保存编译结果和 SHA-256 清单；`changes.patch` 为相对原源码的改动。2026-11-13 已实际烧录并完成写入校验；使用工作台后端连接真实采集板、模拟电机，连续 3 次验证 REBOOT 确认 → READY → 新磁数据，均约 1 秒恢复，且传感器时间戳重置。实体电机未操作。

原始完整 4MB Flash、写入日志与实机验证结果保存在 `../../data/firmware_backups/20261113_153347_c3/`。重新打开工作台连接采集板后，请重新采集空载基线。
