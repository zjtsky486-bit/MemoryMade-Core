"""Persistent spillover storage; never moves or deletes the original installation."""
from __future__ import annotations
import json, os, shutil
from pathlib import Path

FALLBACK_ROOT = Path('D:/1223456789')
try:
    from .storage_policy import reserve_bytes,manual_approval_required,d_reserve_bytes
except ImportError:
    from storage_policy import reserve_bytes,manual_approval_required,d_reserve_bytes
RESERVE = reserve_bytes()

class ModelRoot:
    """Read newly installed D models first, and retain access to existing C models."""
    def __init__(self,original):self.original=Path(original)/'assets/models'
    def __truediv__(self,name):
        if manual_approval_required() and (self.original/name).exists():return self.original/name
        candidate=FALLBACK_ROOT/'assets/models'/name
        return candidate if candidate.exists() else self.original/name

def configure_environment(root,resource):
    root=Path(root);resource=Path(resource)
    os.environ['MEMORYMADE_RESOURCE_ROOT']=str(resource)
    os.environ['MEMORYMADE_WRITE_ROOT']=str(root)
    for name,relative in {'MEMORYMADE_DATA_DIR':'data','GRADIO_TEMP_DIR':'temp/gradio','TEMP':'temp','TMP':'temp',
                          'HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','PIP_CACHE_DIR':'cache/pip',
                          'MPLCONFIGDIR':'cache/matplotlib','YOLO_CONFIG_DIR':'cache/ultralytics'}.items():
        path=root/relative;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'

def below_reserve():
    return os.name == 'nt' and shutil.disk_usage('C:/').free <= RESERVE

def write_root(original=None):
    explicit = os.getenv('MEMORYMADE_WRITE_ROOT')
    if explicit:
        return Path(explicit).resolve()
    if manual_approval_required():
        return Path(original or Path(__file__).resolve().parents[2]).resolve()
    marker = FALLBACK_ROOT / '.memorymade-storage-active.json'
    if below_reserve() or marker.exists() or (FALLBACK_ROOT/'.memorymade-storage-requested.json').exists():
        if not Path('D:/').is_dir():
            raise RuntimeError('C 盘达到 50GB 保护线，但 D 盘不可用，任务已停止。')
        FALLBACK_ROOT.mkdir(parents=True, exist_ok=True)
        return FALLBACK_ROOT
    return Path(original or Path(__file__).resolve().parents[2]).resolve()

def activate_fallback(original):
    original = Path(original).resolve()
    root = FALLBACK_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free <= d_reserve_bytes():
        raise RuntimeError(f'D 盘已达到 {d_reserve_bytes()/1024**3:g} GiB 保护线，停止切换，原文件保留。')
    destination = root / 'MEMORYMADE-Studio'
    # Copy code and UI assets only. Existing models and runtimes remain readable on C.
    shutil.copytree(original / 'MEMORYMADE-Studio', destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('data', '__pycache__', '.venv', '.git'))
    for folder in ('data', 'logs', 'cache', 'temp', 'assets/models', 'downloads', 'environments', 'tools'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / '.memorymade-storage-active.json').write_text(json.dumps({
        'resource_root': str(original), 'write_root': str(root),
        'code_root': str(destination), 'c_reserve_gib': 50,
        'policy': 'Keep using D after activation; retain original C files.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (root/'运行D盘软件.bat').write_text('@echo off\r\n"'+str(original/'runtime/python.exe')+'" "'+str(destination/'storage_launcher.py')+'"\r\n',encoding='utf-8')
    return root
