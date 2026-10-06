"""Scene reconstruction and explicitly inferred mechanical chair reconstruction."""
import json,shutil,subprocess
from pathlib import Path
from .config import EXPORT_DIR,BLENDER_EXE
from .pipeline import ModelResult,new_project_id,load_mesh,mesh_stats,now_stamp
from .disk_guard import run_guarded
from .ai_stack import run_worker

def semantic_route(report):
    import re
    desc=report.get('description',{})
    name=str(desc.get('object_name','')).strip().lower()
    text=' '.join(str(desc.get(k,'')) for k in ['object_name','detection_prompt']).lower()
    if desc.get('depiction_type')=='illustration' and type(desc.get('subject_count')) is int and desc['subject_count']==1:return 'human'
    if any(term in text for term in ['电竞椅','游戏椅','办公椅','转椅','gaming chair','office chair','swivel chair']):return 'chair'
    if name in ['人','男人','女人','男性','女性','男孩','女孩','小孩','人类','模特','运动员','舞者'] or any(term in text for term in ['人物','人像','肖像','全身照','半身照','人体模型','人形','人物雕像']) or re.search(r'\b(person|human|man|woman|boy|girl|portrait|people|character|athlete|dancer|male|female)\b',text):return 'human'
    if any(term in text for term in ['房屋','房子','建筑','住宅','别墅','寺庙','庙宇','城堡','瓦房','房间','室内','house','building','architecture','temple','castle','cottage','room','interior','facade']):return 'scene'
    return 'object'

def _blender(folder,script):
    with (folder/'blender.log').open('w',encoding='utf-8') as log:
        done=run_guarded([BLENDER_EXE,'--background','--python-exit-code','1','--python',str(Path(__file__).with_name(script)),'--',str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if done.returncode or not (folder/'model.blend').exists():raise ValueError('结构 / 场景 Blender 处理未完成，请查看作品日志。')

def advanced_model(paths,title,report,width,route):
    folder=EXPORT_DIR/new_project_id(route);folder.mkdir(parents=True,exist_ok=True)
    for index,path in enumerate(paths):shutil.copy2(path,folder/f'source-{index}{Path(path).suffix}')
    (folder/'recognition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if route=='chair':
        description=report.get('description',{})
        text=str(description).lower()
        gaming=any(term in text for term in ['gaming','电竞','赛车','游戏椅'])
        visible_holes=any(term in text for term in ['hole','opening','孔洞','开孔','镂空'])
        config={'width_mm':float(width),'holes':2 if gaming or visible_holes else 0,'wheels':5,'gaming':gaming,
                'source':'Qwen object/visible-part description + mechanical chair prior',
                'inferred':['脚架分支与脚轮数量默认五组','升降杆、脚轮厚度与连接关系为常见结构推测','孔洞数量、尺寸与位置为结构模板估计' if gaming or visible_holes else '隐藏结构与真实尺寸未确认'],
                'confirmed_description':description}
        (folder/'chair.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        _blender(folder,'blender_chair.py')
        generation={'backend':'Structure-prior chair','holes':config['holes'],'wheels':config['wheels'],'inferred':config['inferred'],'note':'结构优先示意模型，含推测补全；不是照片级精确复刻。'}
    else:
        generation=run_worker('scene',{'images':paths,'width_mm':width,'output_dir':str(folder)},timeout=1800)
        _blender(folder,'blender_ai_refine.py')
    (folder/'generation.json').write_text(json.dumps(generation,ensure_ascii=False,indent=2),encoding='utf-8')
    mesh=load_mesh(folder/'model.stl');stats=mesh_stats(mesh,width,width,0)
    stats.update(backend=generation['backend']+' + Blender',recognition=report,generation=generation,
                 reconstruction_route=route,blend_path=str(folder/'model.blend'),glb_units='meters',stl_obj_units='millimeters',
                 inference_notice=generation.get('note',''),preserve_photo_color=route=='scene')
    return ModelResult(folder.name,title,'结构优先（含推测）' if route=='chair' else '建筑 / 场景可见表面','照片外观' if route=='scene' else '光敏树脂','#E9E4D8',str(folder/'model.glb'),str(folder/'model.stl'),str(folder/'model.obj'),str(folder/'view-0.png'),paths,stats,now_stamp())
