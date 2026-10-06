"""User-selected disk reserve, shared by launcher and workers."""
import json,os
from pathlib import Path

def policy():
    root=Path(os.getenv('MEMORYMADE_RESOURCE_ROOT',str(Path(__file__).resolve().parents[2])))
    path=root/'.memorymade-storage-policy.json'
    if not path.exists():return {'c_reserve_gib':50,'limit_action':'switch_to_d'}
    value=json.loads(path.read_text(encoding='utf-8'))
    if float(value.get('c_reserve_gib',50))<=0:raise ValueError('Invalid disk reserve')
    return value

def reserve_bytes():return int(float(policy().get('c_reserve_gib',50))*1024**3)
def manual_approval_required():return policy().get('limit_action')=='require_user_approval'


def d_reserve_bytes():
    value=float(policy().get("d_reserve_gib",50))
    if value<=0:raise ValueError("Invalid D disk reserve")
    return int(value*1024**3)
