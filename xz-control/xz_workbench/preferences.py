"""Shared, locked settings updates for UI drafts and recorder labels."""
import fcntl
import json
from pathlib import Path
import uuid


def load_settings(path):
    path=Path(path)
    if not path.exists(): return {}
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict): raise ValueError('设置必须是对象')
        return value
    except (ValueError,UnicodeError) as exc:
        raise ValueError('设置文件损坏，已使用默认设置') from exc


def update_settings(path,changes):
    from .catalog import atomic_json
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name(path.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            try: current=load_settings(path)
            except ValueError:
                # Retain the damaged file for recovery instead of silently discarding it.
                path.rename(path.with_name(path.name+'.invalid-'+uuid.uuid4().hex)); current={}
            current.update(changes); atomic_json(path,current)
        finally: fcntl.flock(lock,fcntl.LOCK_UN)
