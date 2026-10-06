"""Local vision core selection. Activation follows a real image smoke check."""
import json
import os
from pathlib import Path

QWEN3_FILES={'Qwen3VL-8B-Instruct-Q4_K_M.gguf':5027784800,
             'mmproj-Qwen3VL-8B-Instruct-F16.gguf':1159029824}


def observation_schema(comparison=False, image_count=None):
    text={'type':'string'}
    faces={'type':'string','enum':['front','left','back','right','top','bottom']}
    def array(item):return {'type':'array','items':item}
    def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    if comparison:
        count={'minItems':image_count,'maxItems':image_count} if image_count else {}
        views={**array({'type':'string','enum':faces['enum']+['unknown']}),**count}
        observations={**array(obj({'observed_faces':array(faces),'elevation':{'type':'string','enum':['eye','high','low','unknown']}})),**count}
        return obj({'object_name':text,'same_object':{'type':'boolean'},'reason':text,'views':views,'view_observations':observations})
    properties={'object_name':text,'detection_prompt':text,
                'depiction_type':{'type':'string','enum':['photo','illustration','sculpture','unknown']},
                'subject_count':{'anyOf':[{'type':'integer','minimum':0,'maximum':9},{'type':'null'}]},
                'shape':text,'colors':array(text),'material_appearance':text}
    for key in ['visible_parts','visible_details','carving_details','uncertain_or_hidden','held_objects']:properties[key]=array(text)
    properties['observed_faces']=array(faces)
    properties['thin_structures']=array(obj({'part':text,'position':text}))
    properties['attachments']=array(obj({'part':text,'connects_to':text,
                                        'evidence':{'type':'string','enum':['visible','occluded','unknown']},
                                        'evidence_note':text}))
    return obj(properties)


def enhanced_core(base):
    base=Path(base);folder=base/'assets/models/qwen3-vl-8b'
    runner=base/'tools/llama-vulkan/llama-server.exe'
    if not runner.is_file() or any(not (folder/name).is_file() or (folder/name).stat().st_size!=size for name,size in QWEN3_FILES.items()):
        return None
    try:enabled=json.loads((folder/'enabled.json').read_text(encoding='utf-8')).get('enabled') is True
    except (OSError,ValueError,AttributeError):enabled=False
    if not enabled and os.getenv('MEMORYMADE_VISION_CORE')!='qwen3':return None
    return {'name':'Qwen3-VL-8B Instruct Q4_K_M','executable':runner,
            'model':folder/'Qwen3VL-8B-Instruct-Q4_K_M.gguf',
            'projector':folder/'mmproj-Qwen3VL-8B-Instruct-F16.gguf',
            'device':'Vulkan0 / NVIDIA GPU','gpu_layers':'99'}
