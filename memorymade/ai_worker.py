"""Isolated GPU workers. The UI runtime never imports torch or transformers."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(os.getenv('MEMORYMADE_RESOURCE_ROOT', str(Path(__file__).resolve().parents[2])))
from storage import ModelRoot
MODELS = ModelRoot(BASE)
os.environ.setdefault('HF_HOME', str(BASE / 'assets' / 'hf-cache'))
os.environ.setdefault('U2NET_HOME', str(MODELS / 'rembg'))


def parse_description(text):
    import re
    text = text.strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('Qwen 未返回结构化描述：' + text[:160])
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict) or not isinstance(result.get('object_name'), str):
        raise ValueError('Qwen 描述缺少 object_name')
    if result['object_name'].strip() in ['人','男人','女人','男性','女性','男孩','女孩','小孩','人物','人像','模特','运动员','舞者']:
        result['detection_prompt']='person'
    if str(result.get('detection_prompt','')).strip().lower() in ['character','anime character','cartoon character','human','human character'] or (result.get('depiction_type')=='illustration' and type(result.get('subject_count')) is int and result['subject_count']==1):
        result['detection_prompt']='person'
    # A single photograph cannot establish the total count of occluded parts.
    # Keep the original claim for review, without presenting it as verified geometry.
    claims=[]
    pattern=re.compile(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+((?:curved\s+|wooden\s+|thin\s+|thick\s+)?(?:legs|supports|handles|wheels|spokes|bars))\b',re.I)
    def sanitize(value):
        if isinstance(value,list):return [sanitize(item) for item in value]
        if not isinstance(value,str):return value
        def replace(match):
            claims.append(match.group(0));return match.group(1)+' (total count unverified)'
        return pattern.sub(replace,value)
    for key in ['shape','visible_parts','visible_details']:result[key]=sanitize(result.get(key,''))
    faces=result.get('observed_faces',[])
    result['observed_faces']=[face for face in faces if face in ['front','left','back','right','top','bottom']] if isinstance(faces,list) else []
    thin=result.get('thin_structures',[])
    if isinstance(thin,list):
        result['thin_structures']=[{**item,'part':str(item.get('part') or item.get('id') or item.get('name') or '细长部件')} if isinstance(item,dict) else {'part':str(item),'position':'待核对'} for item in thin]
    result['attachment_evidence_is_verified']=False
    if claims:
        result['unverified_count_claims']=claims
        hidden=result.get('uncertain_or_hidden',[])
        if not isinstance(hidden,list):hidden=[str(hidden)]
        result['uncertain_or_hidden']=hidden+['部件总数尚未核实，原始语言模型的数量判断已降为待核对信息']
    return result


def describe_transformers(image_path, target=''):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path = MODELS / 'qwen2.5-vl-3b-transformers'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, min_pixels=256*28*28, max_pixels=768*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    ).to(device).eval()
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    prompt = '''Analyze the main foreground object for 3D reconstruction. Return only a JSON object.
Use Chinese descriptions and an English noun for detection_prompt. Required keys:
object_name, detection_prompt, depiction_type (photo/illustration/sculpture/unknown), subject_count (integer or null), shape, colors (array), material_appearance,
visible_parts (array), visible_details (array), carving_details (array), uncertain_or_hidden (array),
observed_faces (array of front/left/back/right/top/bottom; bottom means the physical underside, not lower screen),
held_objects (array), thin_structures (array; one record per observed rope/ribbon/wire with part and visual location, not one generic class),
attachments (array of {part, connects_to, evidence: visible/occluded/unknown}).
Describe only visible evidence. Do not invent hidden geometry or exact dimensions.
Mark material as appearance rather than a verified fact. Be specific about silhouette,
part count where visible, holes, handles, legs, surface markings and proportions.
Distinguish real relief, incised grooves and pierced carvings from flat painted patterns.
For traditional ornaments describe visible motifs (clouds, meanders, lotus petals), placement,
edge ridges and repetition; do not invent motifs where the photograph is blurry.
For houses describe roof pitch, eaves, ridges, wall corners, plinth, doors and window recesses.
For people, anime characters and character drawings use person for detection_prompt. Describe visible pose, head direction, arms, hands, legs, feet, hair, garment silhouette and folds. State whether the whole body is in frame, if more than one person is present, and if limbs overlap or are hidden. Preserve held props and every visible rope, ribbon, streamer, cord and hanging ornament in thin_structures. For each endpoint, describe the visible connection or mark it occluded/unknown in attachments. Do not invent hidden fingers, face features or anatomy. Skin/muscles and printed clothing patterns are not carved relief.'''
    if target:
        prompt += '\nUser request (use only to select the target, never as visual evidence): ' + target[:1000]
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=1100, do_sample=False)
    decoded = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    result = parse_description(decoded)
    result['backend'] = 'Qwen2.5-VL-3B'
    result['device'] = device
    return result


def describe(image_path, target='', comparison=False, image_paths=None):
    from vision_core import enhanced_core
    core=enhanced_core(BASE)
    if core:
        try:return describe_gguf(image_path,target,comparison,core,image_paths)
        except Exception as exc:
            result=describe_gguf(image_path,target,comparison)
            result['fallback_reason']='增强识别核心未完成，本次使用 Qwen2.5：'+str(exc)[:250]
            return result
    if not comparison and all((MODELS / 'qwen2.5-vl-3b-transformers' / name).exists() for name in ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']):
        return describe_transformers(image_path, target)
    return describe_gguf(image_path,target,comparison)


def describe_gguf(image_path, target='', comparison=False, core=None, image_paths=None):
    import base64
    import io
    import socket
    import subprocess
    import time
    import urllib.request
    from PIL import Image, ImageOps
    executable = core['executable'] if core else BASE / 'tools/llama.cpp/llama-server.exe'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    model=core['model'] if core else MODELS/'qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf'
    projector=core['projector'] if core else MODELS/'qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf'
    native_views=bool(core and comparison and image_paths)
    command = [str(executable), '-m', str(model), '--mmproj', str(projector), '--host', '127.0.0.1', '--port', str(port), '-c', '16384' if core else '8192', '-t', '8', '-ngl', core['gpu_layers'] if core else '0', '--parallel', '1', '--image-min-tokens', '512' if native_views else '1024', '--image-max-tokens', '1536' if native_views else '4096' if core else '2048']
    if core:command.extend(['--device','Vulkan0'])
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    url = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                with urllib.request.urlopen(url + '/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Qwen GGUF 服务启动失败')
                time.sleep(.5)
        image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
        image.thumbnail((1536,1536) if core else (1024,1024))
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=98)
        prompt = '''请仔细观察图片中央的主要物体，为三维建模提供可见结构描述。不要只给出类别。
只输出一个 JSON 对象，包含以下键：object_name（中文物体名称）、detection_prompt（简短英文类别）、
depiction_type（photo真实照片/illustration插画立绘/sculpture雕像/unknown）、subject_count（中央主体人物数量整数，无法确定填null）、
shape（用两句话详细描述整体轮廓、上下比例、弯曲程度和开孔）、colors（颜色数组）、
material_appearance（根据图片估计的材质外观）、visible_parts（可见部件数组）、
visible_details（只列看清的表面纹理、边缘、支撑和连接细节，不为凑数量而编造）、
carving_details（可见雕刻数组，区分真实凸起浮雕、凹槽阴刻、镂空、平面印花；描述位置、纹样、轮廓和重复关系，看不清必须注明。不要把污渍或木纹当雕刻）、
uncertain_or_hidden（看不清或被遮挡的结构数组）、
observed_faces（只列实际看见的物体表面，值为front/left/back/right/top/bottom；bottom是物体真实底面，不是图片下半部）、
held_objects（手持物数组）、thin_structures（实际可见的绳子、飘带、电线等细长结构数组）、
attachments（连接关系数组，每项含part、connects_to、evidence；evidence只能是visible/occluded/unknown）。
不要把开孔或支撑误认为把手。不确定的部件名称请用形状描述。不能编造背面、内部、真实尺寸或精确部件数量。
榫卯、胶粘等内部连接工艺通常无法从图片确认，不能作为可见事实，必须放入 uncertain_or_hidden。
建筑要描述屋顶坡面、屋脊、挑檐、墙角、门窗凹凸与台阶；传统纹样要观察云纹、回纹、莲瓣等轮廓，不能因用户提到某个纹样就编造图片里存在。材质只是外观推测，只有 detection_prompt 用英文，其他内容用中文。'''
        prompt+='\n如果主体是人物或二次元角色立绘，detection_prompt 用 person。观察手持玩具、法杖、每条绳子、纸状飘带、饰品和宽袖长发，不要把它们排除在主体之外。细长结构逐条编号，分别写出位置，不把多条绳子或飘带合成一个泛称。逐条记录两端连接位置；可见连接才用visible，连接点遮住用occluded，不知道用unknown。只写“连接角色/身体”太笼统，应指出手、玩具或法杖的具体可见位置；看不到具体位置必须用unknown或occluded。不能把断开的细绳当背景或假称已连好。人物肌肉、皮肤、衣服印花不是雕刻。描述可见姿态、手脚、衣物褶皱和遮挡，不虚构隐藏手指、背面或人体结构。'
        if comparison:
            prompt='''图片是多张照片组成的编号接触表。逐张比较中央物体的轮廓、颜色、纹理和结构，判断是否可能是同一个具体物体的不同角度；同一类别但形状、材质、部件不同不能算同一物体。无法确定时same_object必须是false。只输出JSON：object_name（中文），same_object（布尔），reason（中文比较依据），views（按编号顺序给出front/left/back/right/top/bottom/unknown），view_observations（按编号顺序给出{observed_faces:[front/left/back/right/top/bottom中实际可见表面],elevation:eye/high/low/unknown}）。不要强制把第一张称为front；纯顶视图用top，底部传感器/脚贴一面才是鼠标bottom，不是图片下方。斜上方照片可以观察到顶面及侧面，但方位不确定时必须用unknown，不能编造标准方位。左右是物体拍摄方位，不是图片位置。不要把多个不同物体说成同一个。'''
        if target:
            prompt += '\nUser request for selecting the object, not visual evidence: ' + target[:1000]
        content=[{'type':'text','text':prompt}]
        if native_views:
            content[0]['text']=prompt.replace('图片是多张照片组成的编号接触表。','后面是按序编号的独立照片，请逐张比较。')
            for index,path in enumerate(image_paths[:6],1):
                frame=ImageOps.exif_transpose(Image.open(path)).convert('RGB');frame.thumbnail((768,768))
                frame_stream=io.BytesIO();frame.save(frame_stream,format='JPEG',quality=98)
                content.extend([{'type':'text','text':f'照片 {index}'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(frame_stream.getvalue()).decode()}}])
        else:content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()}})
        payload = {'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 2200 if core else 1500, 'response_format': {'type': 'json_object'}}
        if core:
            from vision_core import observation_schema
            payload['response_format']['schema']=observation_schema(comparison,len(image_paths[:6]) if native_views else None)
        request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
        result = parse_description(body['choices'][0]['message']['content'])
        if not comparison:
            hidden = result.get('uncertain_or_hidden', [])
            if not isinstance(hidden, list):hidden = [str(hidden)]
            result['uncertain_or_hidden'] = [value for value in hidden if value not in ['无', 'none', '']] + ['背面、底面、内部结构和真实尺寸未拍到时属于推测']
        result.update({'backend':core['name'] if core else 'Qwen2.5-VL-3B GGUF Q4_K_M','device':core['device'] if core else 'cpu'})
        if comparison:result['comparison_input']='native images' if native_views else 'contact sheet'
        if core and not comparison:
            from detail_observation import needs_detail_review,review_accessories
            if needs_detail_review(result):
                try:result=review_accessories(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'),result,url)
                except Exception as exc:
                    result['detail_review']={'status':'incomplete','geometry_connections_verified':False,
                                             'reason':str(exc)[:250]}
                    result['uncertain_or_hidden'].append('细长部件局部复核未完成，不能将端点连接判断视为已确认。')
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def detect(image_path, description):
    import numpy as np
    import torch
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from ultralytics import YOLOWorld
    # Keep text weights local after the first installation.
    model = YOLOWorld(str(MODELS / 'yolo-world/yolov8s-worldv2.pt'))
    noun = str(description.get('detection_prompt', '')).strip()[:120]
    names = list(dict.fromkeys([noun] if noun else []))
    if not names:
        names = ['stool', 'chair', 'table', 'vase', 'bottle', 'cup', 'toy', 'statue', 'lamp', 'shoe', 'bag', 'animal']
    # Reuse the existing CLIP text ONNX instead of downloading another 350MB model.
    tokenizer = Tokenizer.from_file(str(BASE / 'MEMORYMADE-Studio/assets/models/clip-vit-base-patch32/tokenizer.json'))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=49407, pad_token='<|endoftext|>')
    tokens = tokenizer.encode_batch(names)
    session = ort.InferenceSession(str(MODELS / 'yolo-world/clip-vit-base-patch32-text.onnx'), providers=['CPUExecutionProvider'])
    features = session.run(['text_embeds'], {'input_ids': np.asarray([item.ids for item in tokens], dtype=np.int64), 'attention_mask': np.asarray([item.attention_mask for item in tokens], dtype=np.int64)})[0]
    features = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    model.model.txt_feats = torch.from_numpy(features.astype(np.float32)).reshape(1, len(names), 512)
    model.model.model[-1].nc = len(names)
    model.model.names = dict(enumerate(names))
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    result = model.predict(source=image, conf=0.15, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
    detections = []
    for box in result.boxes:
        detections.append({'label': names[int(box.cls.item())], 'score': float(box.conf.item()), 'box': box.xyxy[0].cpu().tolist()})
    return sorted(detections, key=lambda item: item['score'], reverse=True)


def ground(image_path, description, include_parts=False):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    path=MODELS/'grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(path,local_files_only=True).to('cuda').eval()
    image=ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    text=str(description.get('detection_prompt','object')).strip().rstrip('.')+'.'
    inputs=processor(images=image,text=text,return_tensors='pt').to('cuda')
    with torch.inference_mode():outputs=model(**inputs)
    result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,box_threshold=.25,text_threshold=.20,target_sizes=[(image.height,image.width)])[0]
    def boxes(result):
        return sorted([{'label':str(label),'score':float(score),'box':box.tolist(),'source':'Grounding DINO'} for label,score,box in zip(result.get('text_labels',result['labels']),result['scores'].cpu(),result['boxes'].cpu())],key=lambda x:x['score'],reverse=True)
    whole=boxes(result)
    if not include_parts:return whole
    parts=description.get('visible_parts',[])
    if not isinstance(parts,list):parts=[]
    translations={'腿':'leg','凳腿':'leg','座面':'seat','靠背':'backrest','扶手':'armrest','底座':'base','底环':'base ring','把手':'handle','轮子':'wheel','脚轮':'wheel','升降杆':'gas lift','孔洞':'hole','屋顶':'roof','屋脊':'roof ridge','柱子':'pillar','门窗':'window','台阶':'steps','绳子':'rope','细绳':'cord','纸流苏':'paper streamer','纸带':'paper streamer','飘带':'ribbon','纸状飘带':'paper streamer','流苏':'tassel','电线':'wire','玩具':'toy','法杖':'staff','权杖':'staff','长杖':'staff','手':'hand'}
    accessory_parts=[]
    for key in ['thin_structures','held_objects']:
        values=description.get(key,[])
        if isinstance(values,list):accessory_parts.extend(value.get('part','') if isinstance(value,dict) else str(value) for value in values)
    parts=accessory_parts+parts
    terms=[]
    for part in parts[:10]:
        label=str(part)
        term=translations.get(label)
        if term is None:term=next((translation for phrase,translation in translations.items() if phrase in label),label)
        term=term.strip()[:45]
        term={'legs':'leg','handles':'handle','wheels':'wheel'}.get(term,term)
        if term and term.isascii():
            candidate=term if term in ['rope','cord','ribbon','paper streamer','tassel','wire','toy','staff'] else text.rstrip('.')+' '+term
            if candidate not in terms:terms.append(candidate)
    part_boxes=[]
    if terms:
        part_inputs=processor(images=image,text=' . '.join(terms)+' .',return_tensors='pt').to('cuda')
        with torch.inference_mode():part_outputs=model(**part_inputs)
        part_result=processor.post_process_grounded_object_detection(part_outputs,part_inputs.input_ids,box_threshold=.30,text_threshold=.20,target_sizes=[(image.height,image.width)])[0]
        part_boxes=boxes(part_result)
    return {'objects':whole,'parts':part_boxes}


def analyze(request):
    result = {'ok': False, 'errors': [], 'detections': []}
    try:
        result['description'] = describe(request['image'], request.get('target', ''))
        result['ok'] = True
    except Exception as exc:
        result['errors'].append('Qwen: ' + str(exc))
    # Explicitly unload the language model before loading detector.
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        torch = None
    try:
        result['detections'] = detect(request['image'], result.get('description', {}))
        result['detector'] = 'YOLO-World v2'
    except Exception as exc:
        result['errors'].append('YOLO-World: ' + str(exc))
    gc.collect()
    if torch is not None and torch.cuda.is_available():torch.cuda.empty_cache()
    if (MODELS/'grounding-dino-tiny/model.safetensors').exists():
        try:
            evidence=ground(request['image'],result.get('description',{}),include_parts=True)
            result['grounded_detections']=evidence['objects']
            result['grounded_parts']=evidence['parts']
            result['grounding_backend']='Grounding DINO tiny'
        except Exception as exc:
            result['errors'].append('Grounding DINO: '+str(exc))
    return result


def reconstruct(request):
    import numpy as np
    import torch
    import trimesh
    from PIL import Image, ImageOps
    import rembg
    sys.path.insert(0, str(BASE / 'tools/TripoSR-main'))
    from tsr.system import TSR
    from tsr.utils import resize_foreground
    folder = Path(request['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(request['image'])).convert('RGBA')
    image.save(folder / 'source.png')
    if request.get('foreground'):
        image=Image.open(request['foreground']).convert('RGBA')
    box = request.get('box')
    if request.get('foreground'):box=None
    if box and len(box) == 4:
        x1, y1, x2, y2 = box
        margin = max(x2-x1, y2-y1) * .08
        crop = (max(0, int(x1-margin)), max(0, int(y1-margin)), min(image.width, int(x2+margin)), min(image.height, int(y2+margin)))
        if crop[2] > crop[0] and crop[3] > crop[1]:
            image = image.crop(crop)
    if image.getextrema()[3][0] == 255:
        session = rembg.new_session('u2net', providers=['CPUExecutionProvider'])
        image = rembg.remove(image, session=session)
    alpha = np.asarray(image)[:, :, 3]
    if int((alpha > 128).sum()) < 100:
        raise ValueError('去背景后没有有效主体，请使用背景简单、主体完整的图片。')
    image.save(folder / 'foreground.png')
    image = resize_foreground(image, .85)
    rgba = np.array(image).astype(np.float32) / 255
    image = Image.fromarray(((rgba[:, :, :3]*rgba[:, :, 3:4]+(1-rgba[:, :, 3:4])*.5)*255).astype(np.uint8))
    image.save(folder / 'input.png')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TSR.from_pretrained(str(MODELS / 'triposr'), config_name='config.yaml', weight_name='model.ckpt')
    model.renderer.set_chunk_size(4096)
    model.to(device).eval()
    with torch.inference_mode():
        codes = model([image], device=device)
        meshes = model.extract_mesh(codes, has_vertex_color=True, resolution=int(request.get('resolution', 256)))
    mesh = meshes[0]
    if len(mesh.faces) < 10:
        raise ValueError('TripoSR 没有生成有效表面。')
    # One uniform scale preserves proportions; UI width determines the longest dimension.
    scale = float(request.get('width_mm', 100)) / max(float(mesh.extents.max()), 1e-6)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    # TripoSR uses Z-up; glTF uses Y-up. Blender reverses this at import.
    gltf_mesh = mesh.copy()
    gltf_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    gltf_mesh.apply_scale(.001)  # glTF specifies meters; STL/OBJ retain millimeters.
    gltf_mesh.export(folder / 'model.glb')
    mesh.export(folder / 'model.stl')
    mesh.export(folder / 'model.obj')
    return {'ok': True, 'glb': str(folder / 'model.glb'), 'stl': str(folder / 'model.stl'), 'obj': str(folder / 'model.obj'), 'device': device, 'backend': 'TripoSR', 'vertices': len(mesh.vertices), 'faces': len(mesh.faces), 'watertight': bool(mesh.is_watertight)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['analyze', 'reconstruct','compare'])
    parser.add_argument('request')
    parser.add_argument('response')
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding='utf-8'))
        result = {'ok':True,'comparison':describe(request['image'],comparison=True,image_paths=request.get('images'))} if args.operation=='compare' else (analyze(request) if args.operation == 'analyze' else reconstruct(request))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {'ok': False, 'error': str(exc)}
    Path(args.response).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    from disk_guard import start_watchdog
    start_watchdog()
    raise SystemExit(main())
