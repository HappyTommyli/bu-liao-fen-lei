#!/usr/bin/env bash
set -euo pipefail
trainer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---cuda}"
case "$mode" in
  --cuda) torch_version='2.14.1+cu130'; torch_index='https://download.pytorch.org/whl/cu130' ;;
  --cpu) torch_version='2.14.1+cpu'; torch_index='https://download.pytorch.org/whl/cpu' ;;
  *) printf '用法：bash 安装依赖.sh [--cuda|--cpu]\n'; exit 2 ;;
esac
python3 -m venv "$trainer_root/.venv"
"$trainer_root/.venv/bin/python" -m pip install --upgrade pip
"$trainer_root/.venv/bin/python" -m pip install --no-cache-dir --retries 12 --timeout 120 "torch==$torch_version" --index-url "$torch_index"
"$trainer_root/.venv/bin/python" -m pip install -r "$trainer_root/requirements.txt"
"$trainer_root/.venv/bin/python" "$trainer_root/train_cloth.py" diagnose-gpu
printf '\n依赖安装完成。CUDA还需NVIDIA系统驱动；Ubuntu可用 ubuntu-drivers devices 查看推荐版本。\n'
printf '运行“启动布料分类训练.sh”打开界面；GPU不可用时不会自动切CPU。\n'
