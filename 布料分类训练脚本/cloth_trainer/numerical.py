"""Numerical boundary checks with compact, JSON-safe diagnostic context."""
import math
import torch
from .recovery import nonfinite_tensors

VERSION='numerical-safety-v2'


class NumericalFailure(ValueError):
    def __init__(self,stage,**details):
        super().__init__(f'训练遇到非有限数值（{stage}）；已中止，未跳过异常批次')
        self.details=dict(stage=stage,**details)


def tensor_summary(value):
    x=torch.as_tensor(value).detach();finite=torch.isfinite(x);valid=x[finite]
    return dict(shape=list(x.shape),nonfinite=int((~finite).sum()),
                min=float(valid.min()) if valid.numel() else None,max=float(valid.max()) if valid.numel() else None)


def trial_details(trials):
    return [dict(id=t.id,path=t.path,sha256=t.sha256,group=t.group,label=t.label,direction=t.direction,
                 samples=len(t.values),input=tensor_summary(t.values)) for t in trials]


def require_finite(value,stage,**details):
    bad=nonfinite_tensors(value)
    if bad:raise NumericalFailure(stage,nonfinite_tensors=bad,**details)


def json_safe(value):
    """Diagnostics themselves must remain writable when their scalars are NaN."""
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value


@torch.no_grad()
def clip_gradients(named_parameters,max_norm=1.):
    """Compute and apply clipping in float64; keep parameter/gradient dtypes."""
    gradients={name:p.grad for name,p in named_parameters if p.grad is not None}
    require_finite(gradients,'gradients')
    if not gradients:raise ValueError('未产生可训练梯度')
    if not math.isfinite(max_norm) or max_norm<=0:raise ValueError('梯度裁剪上限必须为正的有限值')
    norm=float(torch.stack([g.double().norm() for g in gradients.values()]).norm())
    maximum=max(float(g.abs().max()) for g in gradients.values())
    if not math.isfinite(norm):raise NumericalFailure('gradient_norm',gradient_norm=None,max_abs_gradient=maximum)
    coefficient=min(1.,max_norm/(norm+1e-6))
    # Multiply in float64 too: converting a tiny coefficient to float32 first
    # loses precision (or underflows) for otherwise representable results.
    if coefficient<1:
        for g in gradients.values():g.copy_(g.double().mul_(coefficient))
    require_finite(gradients,'clipped_gradients',gradient_norm=norm,max_abs_gradient=maximum,clip_coefficient=coefficient)
    clipped_norm=float(torch.stack([g.double().norm() for g in gradients.values()]).norm())
    if not math.isfinite(clipped_norm) or clipped_norm>max_norm*(1+1e-6):
        raise NumericalFailure('clipped_gradient_norm',gradient_norm=norm,clipped_gradient_norm=clipped_norm,
                               max_abs_gradient=maximum,clip_coefficient=coefficient)
    return dict(norm=norm,max_abs=maximum,coefficient=coefficient,clipped_norm=clipped_norm,clipped=norm>max_norm)
