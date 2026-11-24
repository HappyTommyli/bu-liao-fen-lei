"""Atomic, checksummed epoch state. Separate from validation-best model exports."""
from pathlib import Path
import hashlib
import io
import os
import tempfile
import torch

FORMAT='cloth-epoch-recovery-v1'


def nonfinite_tensors(value,prefix=''):
    """Return locations, never tensor payloads, including nested optimizer state."""
    bad=[]
    if isinstance(value,torch.Tensor):
        if not bool(torch.isfinite(value).all()):bad.append(prefix or 'value')
    elif isinstance(value,dict):
        for key,item in value.items():bad.extend(nonfinite_tensors(item,f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):bad.extend(nonfinite_tensors(item,f'{prefix}[{i}]'))
    return bad


def save_recovery(folder,state,binding):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if nonfinite_tensors(state):raise ValueError('恢复点包含非有限张量，拒绝保存')
    buf=io.BytesIO();torch.save(dict(format=FORMAT,binding=binding,state=state),buf)
    payload=buf.getvalue();header=(FORMAT+' '+hashlib.sha256(payload).hexdigest()+'\n').encode('ascii')
    fd,name=tempfile.mkstemp(prefix='recovery-',suffix='.tmp',dir=folder)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(header);stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(name,folder/'recovery.ckpt')
    finally:
        if os.path.exists(name):os.unlink(name)


def load_recovery(folder,binding,*,allow_runtime_upgrade=False):
    path=Path(folder)/'recovery.ckpt'
    if not path.exists():return None
    try:
        raw=path.read_bytes();header,payload=raw.split(b'\n',1)
        if header!=(FORMAT+' '+hashlib.sha256(payload).hexdigest()).encode('ascii'):
            raise ValueError('恢复点字节校验失败')
        envelope=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
        original=envelope['binding'];upgrade=None
        if original!=binding:
            compatible=(allow_runtime_upgrade and original.get('runtime')=='numerical-safety-v1' and binding.get('runtime')=='numerical-safety-v2'
                        and dict(original,runtime='numerical-safety-v2')==binding)
            if not compatible:raise ValueError('恢复点与冻结配置/数据/模型关联不匹配')
            upgrade=dict(source_sha256=hashlib.sha256(raw).hexdigest(),**{'from':original['runtime'],'to':binding['runtime']})
        if envelope['format']!=FORMAT:raise ValueError('恢复点格式不匹配')
        state=envelope['state']
        if upgrade:state['_runtime_upgrade']=upgrade
        if nonfinite_tensors(state):raise ValueError('恢复点包含非有限张量')
        return state
    except (OSError,ValueError,KeyError,RuntimeError,EOFError) as exc:
        raise ValueError('恢复点损坏或关联校验失败；保留文件并检查后再继续：'+str(exc)) from exc
