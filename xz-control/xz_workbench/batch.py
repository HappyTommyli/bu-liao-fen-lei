"""Preflight repeated recipes without changing the physical origin."""
from .model import finite,plan_steps,validate_capture


def validate_batch(recipe,current,config,count,interval,baseline_available):
    if type(count) is not int or not 1<=count<=10000:
        raise ValueError('批量次数必须是 1–10000 的整数')
    if not 0<=finite(interval,'轮次间隔')<=86400:
        raise ValueError('轮次间隔必须在 0–86400 秒之间')
    if len(recipe.steps)*count>100000: raise ValueError('整批动作超过 100000 步，请减少轮次')
    position=dict(current); valid=baseline_available
    if recipe.config_changed(config): raise ValueError('流程配置已改变，请检查并采用当前配置')
    for round_number in range(1,count+1):
        try:
            planned=plan_steps(recipe,position,config); validate_capture(recipe,valid)
            for step in planned:
                if step.command and step.duration*2+5>600: raise ValueError('单步超过运动超时上限')
                if step.kind=='restart_sensor': valid=False
                elif step.kind=='baseline': valid=True
            position.update(x=planned[-1].x,z=planned[-1].z)
        except ValueError as exc: raise ValueError(f'第 {round_number} 轮：{exc}') from exc
