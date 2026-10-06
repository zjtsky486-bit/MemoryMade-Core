from __future__ import annotations

import html
import json
import os
import subprocess
import threading
import uuid
import hashlib
import time
from pathlib import Path

from .config import APP_ROOT, EXPORT_DIR
from .disk_guard import ensure_disk_space, run_guarded
from .storage import ModelRoot, write_root
from .recognition_cache import RecognitionCache

BASE = Path(os.getenv('MEMORYMADE_RESOURCE_ROOT', str(APP_ROOT.parent)))
VISION_PYTHON = BASE / 'environments/vision/Scripts/python.exe'
TRIPO_PYTHON = BASE / 'environments/reconstruction/Scripts/python.exe'
_hy21_root = BASE if (BASE/'tools/Hunyuan3D-2.1/hy3dshape/hy3dshape/pipelines.py').exists() else Path('D:/1223456789')
HUNYUAN21_PYTHON = _hy21_root/'environments/hunyuan21/Scripts/python.exe'
HUNYUAN21_TOOLS = _hy21_root/'tools/Hunyuan3D-2.1'
HUNYUAN21_MODELS = _hy21_root/'assets/models/hunyuan3d-2.1'
_spill=write_root(BASE)
if (_spill/'environments/vision/Scripts/python.exe').exists():VISION_PYTHON=_spill/'environments/vision/Scripts/python.exe'
if (_spill/'environments/reconstruction/Scripts/python.exe').exists():TRIPO_PYTHON=_spill/'environments/reconstruction/Scripts/python.exe'
if (_spill/'environments/hunyuan21/Scripts/python.exe').exists():HUNYUAN21_PYTHON=_spill/'environments/hunyuan21/Scripts/python.exe'
MODEL_ROOT = ModelRoot(BASE)
HUMAN_PYTHON=BASE/'environments/human/Scripts/python.exe'
GPU_LOCK = threading.Lock()
ANALYSIS_CACHE = RecognitionCache(32)


def stack_status():
    from .vision_core import enhanced_core
    core=enhanced_core(BASE)
    qwen = MODEL_ROOT / 'qwen2.5-vl-3b-transformers'
    hunyuan_source=False
    try:
        manifest=json.loads((MODEL_ROOT/'upgrade-sources.json').read_text(encoding='utf-8'))
        source=BASE/'tools'/('Hunyuan3D-2-'+manifest['hunyuan_source']['commit'])
        hunyuan_source=(source/'hy3dgen/shapegen/pipelines.py').exists()
    except (OSError,ValueError,KeyError):pass
    return {
        'vision_core':core['name'] if core else 'Qwen2.5-VL-3B',
        'human': HUMAN_PYTHON.exists() and all((BASE/'assets/models/mediapipe-human'/name).exists() for name in ['pose_landmarker_heavy.task','face_landmarker.task','hand_landmarker.task']),
        'qwen': VISION_PYTHON.exists() and (bool(core) or ((qwen / 'model.safetensors.index.json').exists() and all((qwen / name).exists() for name in ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'])) or ((BASE / 'tools/llama.cpp/llama-server.exe').exists() and all((MODEL_ROOT / 'qwen2.5-vl-3b' / name).exists() for name in ['Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf', 'mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf']))),
        'world': VISION_PYTHON.exists() and all((MODEL_ROOT / 'yolo-world' / name).exists() for name in ['yolov8s-worldv2.pt', 'clip-vit-base-patch32-text.onnx']),
        'triposr': TRIPO_PYTHON.exists() and all((MODEL_ROOT / 'triposr' / name).exists() for name in ['model.ckpt', 'config.yaml', 'dino_config.json']) and (MODEL_ROOT / 'rembg/u2net.onnx').exists(),
        'birefnet': VISION_PYTHON.exists() and all((MODEL_ROOT/'birefnet-lite'/name).exists() for name in ['model.safetensors','config.json','birefnet.py','BiRefNet_config.py']),
        'grounding_dino': VISION_PYTHON.exists() and all((MODEL_ROOT/'grounding-dino-tiny'/name).exists() for name in ['model.safetensors','config.json','preprocessor_config.json','tokenizer.json']),
        'hunyuan': VISION_PYTHON.exists() and hunyuan_source and all((MODEL_ROOT/'hunyuan3d-mini/hunyuan3d-dit-v2-mini'/name).exists() for name in ['model.fp16.safetensors','config.yaml']),
        'hunyuan21': HUNYUAN21_PYTHON.exists() and (HUNYUAN21_TOOLS/'hy3dshape/hy3dshape/pipelines.py').exists() and (HUNYUAN21_MODELS/'hunyuan3d-dit-v2-1/model.fp16.ckpt').exists(),
        'hunyuan21_pbr_models': HUNYUAN21_PYTHON.exists() and (HUNYUAN21_MODELS/'hunyuan3d-dit-v2-1/model.fp16.ckpt').exists() and (HUNYUAN21_MODELS/'hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin').exists() and (MODEL_ROOT/'dinov2-giant/model.safetensors').exists() and (MODEL_ROOT/'realesrgan/RealESRGAN_x4plus.pth').exists(),
        'hunyuan21_pbr': HUNYUAN21_PYTHON.exists() and (HUNYUAN21_TOOLS/'hy3dshape/hy3dshape/pipelines.py').exists() and (HUNYUAN21_MODELS/'hunyuan3d-dit-v2-1/model.fp16.ckpt').exists() and (HUNYUAN21_MODELS/'hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin').exists() and (MODEL_ROOT/'dinov2-giant/model.safetensors').exists() and (MODEL_ROOT/'realesrgan/RealESRGAN_x4plus.pth').exists() and any((HUNYUAN21_PYTHON.parent.parent/'Lib/site-packages').glob('custom_rasterizer_kernel*.pyd')) and any((HUNYUAN21_PYTHON.parent.parent/'Lib/site-packages').glob('mesh_inpaint_processor*.pyd')),
        'material_siglip2': VISION_PYTHON.exists() and all((MODEL_ROOT/'siglip2-material'/name).exists() for name in ['model.safetensors','config.json','tokenizer.json']),
        'scene_vggt': VISION_PYTHON.exists() and all((MODEL_ROOT/'vggt'/name).exists() for name in ['model.safetensors','source-code.json']),
    }


def run_worker(operation, payload, timeout=900):
    ensure_disk_space('AI 识别与建模')
    folder = Path(os.getenv('MEMORYMADE_WRITE_ROOT', str(BASE))) / 'logs' / 'ai' / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    request, response = folder / 'request.json', folder / 'response.json'
    request.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    if operation in ('human','human_foreground'):
        python=HUMAN_PYTHON
    elif operation == 'reconstruct':
        python = TRIPO_PYTHON
    elif operation in ('hunyuan21', 'hunyuan21_pbr'):
        python = HUNYUAN21_PYTHON
    else:
        python = VISION_PYTHON
    if not python.exists():
        raise RuntimeError('AI 独立环境尚未安装：' + str(python))
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env.setdefault('HF_HOME', str(BASE / 'assets/hf-cache'))
    env['U2NET_HOME'] = str(MODEL_ROOT / 'rembg')
    queued_at = time.perf_counter()
    with GPU_LOCK, (folder / 'worker.log').open('w', encoding='utf-8') as log:
        started_at = time.perf_counter()
        worker='human_worker.py' if operation in ('human','human_foreground') else 'visual_worker.py' if operation in ['material','scene'] else ('audio_worker.py' if operation=='listen' else ('precision_worker.py' if operation in ['segment','hunyuan','hunyuan21','hunyuan21_pbr'] else 'ai_worker.py'))
        cwd = str(HUNYUAN21_TOOLS) if operation in ('hunyuan21','hunyuan21_pbr') else str(APP_ROOT)
        completed = run_guarded([str(python), str(Path(__file__).with_name(worker)), operation, str(request), str(response)], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (folder / 'timing.json').write_text(json.dumps({'operation':operation,
        'queue_seconds':round(started_at-queued_at,3),
        'execution_seconds':round(time.perf_counter()-started_at,3)}),encoding='utf-8')
    if not response.exists():
        raise RuntimeError('AI 工作进程未完成，请查看 ' + str(folder / 'worker.log'))
    result = json.loads(response.read_text(encoding='utf-8'))
    if completed.returncode or not result.get('ok'):
        raise RuntimeError(result.get('error') or '; '.join(result.get('errors', [])) or 'AI 推理失败')
    return result


def _analysis_signature(core_name):
    # Invalidate on worker/prompt changes, model replacements and core activation.
    files = [Path(__file__), Path(__file__).with_name('ai_worker.py'),
             Path(__file__).with_name('vision_core.py'), Path(__file__).with_name('detail_observation.py')]
    for directory in ('qwen3-vl-8b','qwen2.5-vl-3b','qwen2.5-vl-3b-transformers',
                      'yolo-world','grounding-dino-tiny'):
        files.extend(sorted((MODEL_ROOT/directory).glob('*')))
    signature = []
    for file in files:
        if file.is_file():
            stat = file.stat()
            signature.append((str(file), stat.st_size, stat.st_mtime_ns))
    return (core_name, os.getenv('MEMORYMADE_VISION_CORE',''), tuple(signature))


def analyze_object(path, target=''):
    status = stack_status()
    if not status['qwen']:
        return {'ok': False, 'errors': ['Qwen 模型或环境未就绪'], 'detections': []}
    try:
        ensure_disk_space('图片识别')
        source = Path(path).resolve()
        before = source.stat()
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        key = (digest, target, _analysis_signature(status['vision_core']))
        def compute():
            result = run_worker('analyze', {'image':str(source),'target':target})
            after = source.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise RuntimeError('识别期间照片发生变化，请重新上传。')
            return result
        return ANALYSIS_CACHE.get(key, compute)
    except Exception as exc:
        return {'ok': False, 'errors': [str(exc)], 'detections': []}




def generate_local_3d(paths, title, subtitle, story, width_mm, material, color, quality='标准（256）', backend='triposr',views=None,recognition=None,subject_kind='object'):
    from .pipeline import ModelResult, create_preview_card, load_mesh, mesh_stats, new_project_id, now_stamp
    if len(paths) != 1 and not views:
        raise ValueError('此三维模式每次重建一个物体，请上传一张主视图；多张图不会自动融合。')
    status_key = backend if backend in ('hunyuan','hunyuan21','hunyuan21_pbr') else 'triposr'
    status = stack_status()
    if not status.get(status_key):
        raise RuntimeError('所选三维模型或环境尚未就绪，请先运行安装脚本。')
    primary=views.get('front',paths[0]) if views else paths[0]
    report = recognition or analyze_object(primary, story)
    from .advanced_reconstruction import semantic_route
    if subject_kind=='object' and semantic_route(report)=='human':subject_kind='human'
    if subject_kind=='human' and backend not in ('hunyuan','hunyuan21','hunyuan21_pbr'):
        backend='hunyuan21' if status.get('hunyuan21') else 'hunyuan'
        if not status.get(backend):raise RuntimeError('人物精修所需的混元模型未就绪。')
    if subject_kind=='human' and not status.get('human'):
        raise RuntimeError('人物识别组件未就绪，请检查 D 盘人物模型文件。')
    detections = report.get('detections', [])
    # Prefer detector only when it has adequate confidence. Keep whole image otherwise.
    grounded=report.get('grounded_detections',[])
    box = grounded[0]['box'] if grounded and grounded[0]['score'] >= .30 else (detections[0]['box'] if detections and detections[0]['score'] >= .25 else None)
    project_id = new_project_id(backend)
    folder = EXPORT_DIR / project_id
    from .fidelity import QUALITY_PROFILES
    profile=QUALITY_PROFILES.get(quality,QUALITY_PROFILES['精细（384）'])
    profile=dict(profile)
    if subject_kind=='human':
        profile.update(resolution=max(512,profile['resolution']),steps=max(80,profile['steps']),num_chunks=min(2000,profile['num_chunks']))
        thin=report.get('description',{}).get('thin_structures',[])
        if isinstance(thin,list) and thin:
            profile.update(resolution=max(640,profile['resolution']),steps=max(100,profile['steps']))
            profile['detail_reason']='照片识别到绳子、飘带等细长结构，人物采用至少640采样和100步；不能保证遮挡连接已恢复。'
    resolution=profile['resolution']
    segmentation=None
    payload={'image':primary,'box':box if primary==paths[0] else None,'output_dir':str(folder),'width_mm':float(width_mm),'color':color,**profile}
    folder.mkdir(parents=True,exist_ok=True)
    # Save observational evidence before GPU work so a rejected mesh remains reviewable.
    folder.joinpath('recognition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if report.get('observation_review'):
        folder.joinpath('observation-review.json').write_text(json.dumps(report['observation_review'],ensure_ascii=False,indent=2),encoding='utf-8')
    if subject_kind=='human':
        human=run_worker('human',{'image':primary,'output_dir':str(folder),'observation':report.get('description',{})},timeout=300)
        payload['human_report']=str(folder/'human-recognition.json')
        payload['subject_kind']='human'
    if status['birefnet']:
        # Multiview conditioning already uses the whole foreground for every angle.
        segmentation=run_worker('segment',{'image':primary,'box':None if views or subject_kind in ['building','chair','human'] else payload['box'],'output':str(folder/'foreground.png')},timeout=300)
        if subject_kind=='human':
            segmentation=run_worker('human_foreground',{'foreground':segmentation['foreground'],'report':payload['human_report'],'output':str(folder/'human-foreground.png')},timeout=300)
            if report.get('description',{}).get('detail_review'):
                from .detail_observation import check_endpoints_against_foreground
                from PIL import Image,ImageOps
                with Image.open(primary) as original_image:original_size=ImageOps.exif_transpose(original_image).size
                report['description']=check_endpoints_against_foreground(report['description'],folder/'foreground.png',original_size)
                if report.get('observation_review'):
                    report['observation_review'].update(attachments=report['description'].get('attachments',[]),thin_structures=report['description'].get('thin_structures',[]))
                    folder.joinpath('observation-review.json').write_text(json.dumps(report['observation_review'],ensure_ascii=False,indent=2),encoding='utf-8')
                folder.joinpath('recognition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        payload['foreground']=segmentation['foreground']
    if views:
        payload['views']={}
        import shutil
        for index,(tag,path) in enumerate(views.items()):
            shutil.copy2(path,folder/f'source-{tag}{Path(path).suffix}')
            if segmentation and Path(path).resolve() == Path(primary).resolve():
                payload['views'][tag]=segmentation['foreground']
                if subject_kind=='human':
                    view_folder=folder/('human-view-'+tag)
                    view_folder.mkdir(parents=True,exist_ok=True)
                    human_source=Path(payload['human_report'])
                    if human_source.exists():shutil.copy2(human_source,view_folder/'human-recognition.json')
                continue
            foreground=run_worker('segment',{'image':path,'output':str(folder/f'foreground-{tag}.png')},timeout=300)
            if subject_kind=='human':
                view_folder=folder/('human-view-'+tag)
                run_worker('human',{'image':path,'output_dir':str(view_folder),'observation':report.get('description',{})},timeout=300)
                foreground=run_worker('human_foreground',{'foreground':foreground['foreground'],'report':str(view_folder/'human-recognition.json'),'output':str(folder/f'human-foreground-{tag}.png')},timeout=300)
            payload['views'][tag]=foreground['foreground']
        (folder/'view-map.json').write_text(json.dumps(views,ensure_ascii=False,indent=2),encoding='utf-8')
        if subject_kind=='human' and 'front' in views:
            # The generated mesh is front-facing; compare it to the actual front view,
            # not whichever photo happened to be uploaded first.
            payload['foreground']=payload['views']['front']
            payload['human_report']=str(folder/'human-view-front'/'human-recognition.json')
    if backend in ('hunyuan','hunyuan21','hunyuan21_pbr') and not segmentation:
        raise RuntimeError('Hunyuan 重建需要 BiRefNet 前景，请先安装前景模型。')
    result = run_worker(backend if backend in ('hunyuan','hunyuan21','hunyuan21_pbr') else 'reconstruct',payload,timeout=14400)
    report['segmentation']=segmentation
    report['reconstruction_subject_kind']=subject_kind
    folder.joinpath('generation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    folder.joinpath('recognition.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if backend == 'hunyuan21_pbr':
        folder.joinpath('pbr.json').write_text(json.dumps({'backend':'Hunyuan3D-2.1-PBR'}, ensure_ascii=False), encoding='utf-8')
    # Dedicated Blender pass preserves vertex colors and all disconnected components.
    from .config import BLENDER_EXE
    import shutil
    for key in ['glb', 'stl', 'obj']:
        shutil.copy2(result[key], folder / ('raw.' + key))
    if not BLENDER_EXE or not Path(BLENDER_EXE).exists():
        raise RuntimeError('未找到本地 Blender，原始模型已保存：' + str(folder))
    with (folder / 'blender.log').open('w', encoding='utf-8') as log:
        completed = run_guarded([BLENDER_EXE, '--background', '--python-exit-code','1', '--python', str(Path(__file__).with_name('blender_ai_refine.py')), '--', str(folder)], stdout=log, stderr=subprocess.STDOUT, timeout=1800, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if completed.returncode or not (folder / 'model.blend').exists():
        raise RuntimeError('Blender 处理失败，原始模型仍在：' + str(folder))
    for _key, _name in [('glb','model.glb'),('stl','model.stl'),('obj','model.obj')]:
        _path = folder / _name
        if _path.exists(): result[_key] = str(_path)
    # Printable STL has shared geometric vertices; glTF splits sharp-edge normals.
    mesh_mm = load_mesh(result['stl'])
    stats = mesh_stats(mesh_mm, float(width_mm), float(width_mm), 0)
    stats['resolution'] = ' × '.join(str(value) for value in stats['size_mm']) + ' mm'
    stats['components'] = len(mesh_mm.split(only_watertight=False))
    stats['glb_units'] = 'meters'
    stats['stl_obj_units'] = 'millimeters'
    stats['subject_kind']=subject_kind
    if subject_kind=='human':
        from .human_mesh import clean_human_mesh
        desc=report.get('description',{})
        _,final_review=clean_human_mesh(mesh_mm,protect_accessories=bool(desc.get('thin_structures') or desc.get('held_objects')))
        if final_review['removed_small_floaters']:raise ValueError('Blender 导出后出现新的分离碎片，已暂停交付。')
        stats['human_review']=result.get('human_review',{})
        stats['inference_notice']='人物前景保留连在主体上的手持物和细绳；分离的细长部件会暂停交付。背面、遮挡手指、绳子端点仍需旋转复核，网格连通不代表连接位置正确。'
    stats['detail_sampling_mm']=round(float(width_mm)/resolution,4)
    if profile.get('detail_reason'):stats['detail_reason']=profile['detail_reason']
    stats['source_silhouette_holes']=(segmentation or {}).get('holes',0)
    stats['print_review']='网格封闭，可继续检查薄壁、悬空和最小特征。' if stats['watertight'] else '网格未封闭，需要修复后打印。'
    stats.update({'backend': result['backend']+' + Blender', 'recognition': report, 'blend_path': str(folder / 'model.blend'), 'single_image_uncertainty': not bool(views), 'color_source': result.get('color_source','TripoSR vertex colors'), 'mc_resolution': resolution,'segmentation':segmentation,'generation':result})
    preview = folder / 'view-0.png'
    if not preview.exists():
        preview = create_preview_card(paths, folder / 'preview.jpg', title, subtitle, story, color)
    mode_label = {'hunyuan':'AI 精细三维（Hunyuan3D）','hunyuan21':'AI 精细三维（Hunyuan3D 2.1）','hunyuan21_pbr':'AI 精细三维 + PBR（Hunyuan3D 2.1）'}.get(backend, 'AI 完整三维（TripoSR）')
    if subject_kind=='human':mode_label='人物精修（混元 + 人体检查）'
    return ModelResult(project_id, title, mode_label, material if backend.startswith('hunyuan') else '照片原色', color, result['glb'], result['stl'], result['obj'], str(preview), paths, stats, now_stamp())

