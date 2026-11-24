"""Explicit training devices and reproducible runtime identity."""
import os
import platform
import random
import numpy as np
import torch


def diagnostics():
    available=torch.cuda.is_available()
    return dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,cuda_available=available,
                gpu=torch.cuda.get_device_name(0) if available else None,
                cudnn=torch.backends.cudnn.version(),python=platform.python_version())


def resolve_device(name):
    if name not in ('cpu','cuda'):raise ValueError('设备必须为cuda或cpu')
    if name=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用：请安装NVIDIA驱动及CUDA版PyTorch并重启；或明确选择CPU验证。'+str(diagnostics()))
    return torch.device(name)


def seed_runtime(seed,threads):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False


def rng_state(generator):
    ns=np.random.get_state()
    return dict(python=random.getstate(),numpy=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=generator.get_state())


def restore_rng(state,generator):
    random.setstate(state['python']);ns=state['numpy']
    np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    torch.set_rng_state(state['cpu']);generator.set_state(state['loader'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
