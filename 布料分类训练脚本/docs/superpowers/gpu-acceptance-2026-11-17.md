# GPU训练环境与重启后验收

2026-11-17，用户自行重启后，在原脚本目录进行验收。未启动正式数据集训练。

## 已生效环境

- NVIDIA GeForce RTX 3060 Laptop GPU，6 GB 显存。
- NVIDIA驱动595.99.02；DKMS在当前内核6.8.0-149-generic上已安装。
- PyTorch 2.14.1+cu130，CUDA运行库13.0，cuDNN 9.24，Python3.10.12。
- `nvidia-smi`正常，`torch.cuda.is_available()`为true，`pip check`无依赖冲突。

驱动工具显示的CUDA13.2是驱动支持能力；PyTorch实际使用其安装的CUDA13.0运行库。此前下载中断没有跳过校验：全部CUDA wheel通过SHA256校验，缺失的7个驱动依赖通过APT记录的SHA256校验。cuda-pathfinder改用支持Python3.10的1.8.3。下载脚本和实时进度窗口是临时工具，重启后无需继续运行。

## 真实GPU验证

`tests/test_modern_training.py`：16项通过，0跳过，包括GRU、TCN、CERA、Conv-GRU和Conv-CERA的实际CUDA正反向、优化更新与恢复点保存。

额外有限验收使用2个DataLoader进程，在真实GPU上分别检查FP32、AMP FP16、AMP BF16。GRU中断后恢复与连续训练的最终权重逐项完全一致。CPU控制流模拟的OOM重试测试也保留，避免将该测试误称为真实GPU显存不足验证。

从已有冻结数据集只读取两类、每类两次采集，按既有接触段处理生成训练和验证单程。GRU、TCN、CERA分别只运行一轮，输入长度5316/5372点、宽度64、FP32、有效批16、自动微批16；全部实际运行于CUDA。最大已分配显存（含最长序列预检）分别约297MB、573MB、1481MB。这是有限功能验收，不代表分类精度、全量耗时或所有数据上的最大显存需求。

## 数据与交付约束

重启前记录的1045个原实验文件SHA256全部一致，未改变原始数据、旧快照或旧模型。正式实验模型库没有新增此次验收任务。

界面只使用GPU训练，已去除GPU/CPU和精度选择框；默认FP32。CPU保留为内部测试及旧模型兼容用途。旧格式保持原有读取方式；新实验采用开发集五折、选定配置后全开发集重训最终模型、固定内部测试的流程。

用户授权安装驱动，随后决定自行重启；未执行自动重启。代码按已有授权直接修改原应用目录，保留本地`codex/cloth-gpu-standard-pipeline`分支，不合并、不推送。

详细证据在`outputs/verification/gpu-standard/`：

- `device-after-reboot.json`
- `cuda-tests-after-reboot.log`
- `actual-gpu-acceptance.json`及同名日志
- `legacy-integrity-after-reboot.json`
- `full-suite-after-reboot.log`

独立只读代码审查没有Critical或Minor问题；唯一Important问题为AMP显存预检重试状态，已用失败测试复现并修复，后续回归与实际GPU训练验证通过。

完整回归：` .venv/bin/python -m pytest -q --tb=short`，187项通过、0跳过、0失败，71.24秒。未增加正式模型库任务。
