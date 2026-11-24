#!/usr/bin/env bash
set -euo pipefail
trainer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$trainer_root/.venv/bin/python" ]]; then
    printf '请先运行：%s/安装依赖.sh\n' "$trainer_root" >&2
    exit 2
fi
exec "$trainer_root/.venv/bin/python" "$trainer_root/train_cloth.py" "$@"
