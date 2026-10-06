"""Photo-driven routing and explicit shape quality profiles."""
QUALITY_PROFILES={
 '快速（192）':{'resolution':192,'steps':20,'num_chunks':4000},
 '标准（256）':{'resolution':256,'steps':30,'num_chunks':4000},
 '精细（384）':{'resolution':384,'steps':50,'num_chunks':4000},
 '极精细（512 · 80步）':{'resolution':512,'steps':80,'num_chunks':4000},
}
QUALITY_PROFILES['人物精修（640 · 100步）']={'resolution':640,'steps':100,'num_chunks':2000}
DEFAULT_QUALITY='极精细（512 · 80步）'
BUILDING_MODE='建筑实体模型（AI补全）'
ORNAMENTS=['保留照片雕刻（不新增）','回纹（新增浅雕）','祥云纹（新增浅雕）','莲瓣纹（新增浅雕）']
def resolve_route(detected,mode):
    if mode=='建筑 / 场景可见表面':return 'scene'
    if mode=='结构优先（允许推测补全）':return 'chair'
    if mode=='人物精修（人体检查）' or (detected=='human' and mode in ['自动选择','照片外观优先']):return 'human'
    if mode==BUILDING_MODE or (mode=='自动选择' and detected=='scene'):return 'building'
    return 'object'
def production_recommendation(material_report,route):
    from .materials_ai import recommendation
    if route=='human':return '光敏树脂',''
    if route=='building':return '尼龙',''
    choice,custom=recommendation(material_report)
    if custom in ['玻璃','水','植被','未知表面','待确认材料']:return '光敏树脂',''
    return choice,custom
