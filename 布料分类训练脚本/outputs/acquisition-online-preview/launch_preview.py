#!/usr/bin/env python3
"""Manual launcher for the independent preview. --help never imports Qt/devices."""
import argparse
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='独立在线转发预览副本。实物运行前，先关闭原采集程序，避免串口冲突。')
    parser.add_argument('--demo',action='store_true',help='仅模拟设备，不连接硬件')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    os.environ['XZ_WORKBENCH_SETTINGS']=str(root/'preview-runtime'/'settings.json')
    from xz_workbench.ui import main as run
    return run(demo=args.demo)


if __name__=='__main__': raise SystemExit(main())
