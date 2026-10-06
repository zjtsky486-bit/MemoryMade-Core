"""CPU-isolated MediaPipe body/face/hand evidence and person-guided foreground."""
import sys,os,json,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageFilter
BASE=Path(os.getenv('MEMORYMADE_RESOURCE_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(Path(__file__).parent))
from disk_guard import start_watchdog

def inspect_person(request):
    image=ImageOps.exif_transpose(Image.open(request['image'])).convert('RGB')
    observation=request.get('observation',{})
    if observation.get('depiction_type')=='illustration':
        if type(observation.get('subject_count')) is not int or observation['subject_count']!=1:
            raise ValueError('角色立绘未确认单一人物，请使用仅含一个角色、主体清晰的图片。')
        result={'ok':True,'backend':'Illustrated character / foreground and mesh checks',
                'image_size':list(image.size),'poses':[],'faces':[],'hands':[],
                'kind':'illustrated_character','anatomy_verified':False,
                'warnings':['角色立绘不使用真人骨架作为裁切依据；背面、手指和被遮挡的飘带连接仍需复核。']}
        folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
        (folder/'human-recognition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    model=BASE/'assets/models/mediapipe-human'
    small=image.copy();small.thumbnail((1600,1600))
    data=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(small))
    result={'ok':True,'backend':'MediaPipe Heavy Pose + Face + Hands','image_size':list(image.size),'poses':[],'faces':[],'hands':[],'warnings':[]}
    with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model/'pose_landmarker_heavy.task')),num_poses=4,output_segmentation_masks=True)) as task:
        poses=task.detect(data)
    if len(poses.pose_landmarks)>1:raise ValueError('检测到多个人物。请上传仅包含同一个主体的人物照片，避免把不同人的肢体混入模型。')
    for points,world in zip(poses.pose_landmarks,poses.pose_world_landmarks):
        result['poses'].append({'landmarks':[{'index':i,'x':float(p.x),'y':float(p.y),'z':float(p.z),'visibility':float(p.visibility),'presence':float(p.presence)} for i,p in enumerate(points)],'world_landmarks':[{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in world]})
    with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model/'face_landmarker.task')),num_faces=3)) as task:
        faces=task.detect(data)
    if len(faces.face_landmarks)>1:raise ValueError('检测到多张人脸，请上传仅有一个人物主体的照片。')
    for points in faces.face_landmarks:
        result['faces'].append({'landmarks':[{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in points]})
    if not result['poses'] and len(result['faces'])!=1:
        raise ValueError('未能可靠定位单个人物。请提供清晰的全身或半身照片，保留头部、手脚并减少遮挡。')
    if not result['poses']:
        result['kind']='portrait';result['warnings'].append('仅确认脸部，人像以半身或胸像生成，不声称补全真实的下半身。')
    else:
        points=result['poses'][0]['landmarks']
        feet=[points[i] for i in [27,28,31,32]]
        result['kind']='full_body' if all(p['visibility']>.60 and 0<p['y']<.985 for p in feet) else 'partial_body'
        if result['kind']=='partial_body':result['warnings'].append('部分脚部或肢体不可见，缺失部分只能由生成模型推测。')
    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model/'hand_landmarker.task')),num_hands=2)) as task:
        hands=task.detect(data)
    for points in hands.hand_landmarks:
        result['hands'].append({'landmarks':[{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in points]})
    if not result['faces']:result['warnings'].append('脸部不可可靠观察，不能验证五官相似度。')
    if len(result['hands'])<2:result['warnings'].append('部分手指不可可靠观察，不能验证隐藏手指结构。')
    if result['poses'] and poses.segmentation_masks:
        mask=np.squeeze(poses.segmentation_masks[0].numpy_view());path=Path(request['output_dir'])/'human-pose-mask.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.uint8(np.clip(mask,0,1)*255)).resize(image.size,Image.Resampling.BILINEAR).save(path)
        result['pose_mask']=str(path)
    folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    (folder/'human-recognition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def foreground(request):
    from human_foreground import select_person_foreground
    report=json.loads(Path(request['report']).read_text(encoding='utf-8'))
    image=Image.open(request['foreground']).convert('RGBA');alpha=np.asarray(image.getchannel('A'))
    if report.get('pose_mask'):
        pose=Image.open(report['pose_mask']).convert('L').resize(image.size)
        alpha,review=select_person_foreground(alpha,np.asarray(pose))
        report['foreground_review']=review
        if review['retained_nearby_regions']:
            report.setdefault('warnings',[]).append('已保留人物附近可能属于玩具、绳子或饰品的前景；附近背景仍需复核，未凭空补画连接。')
    else:
        report['foreground_review']={'policy':'preserve_illustrated_character_alpha' if report.get('kind')=='illustrated_character' else 'preserve_portrait_alpha','connection_verified':False,'synthetic_connections_added':False}
    image.putalpha(Image.fromarray(alpha));bbox=image.getchannel('A').getbbox()
    if not bbox:raise ValueError('人物前景分离失败。')
    margin=max(3,int(max(bbox[2]-bbox[0],bbox[3]-bbox[1])*.035))
    bounds=(max(0,bbox[0]-margin),max(0,bbox[1]-margin),min(image.width,bbox[2]+margin),min(image.height,bbox[3]+margin))
    report['foreground_landmarks']=[{**p,'x':(p['x']*image.width-bounds[0])/(bounds[2]-bounds[0]),'y':(p['y']*image.height-bounds[1])/(bounds[3]-bounds[1])} for p in (report.get('poses') or [{}])[0].get('landmarks',[])]
    report['foreground_crop']=list(bounds)
    # Check high-confidence visible joints before any shape generation starts.
    misses=[]
    for p in (report.get('poses') or [{}])[0].get('landmarks',[]):
        if p['index'] not in [11,12,13,14,15,16,23,24,25,26,27,28] or p['visibility']<.80:continue
        x=int(p['x']*image.width);y=int(p['y']*image.height);r=max(4,int(max(image.size)*.02))
        if 0<=x<image.width and 0<=y<image.height and not (alpha[max(0,y-r):y+r+1,max(0,x-r):x+r+1]>40).any():misses.append(p['index'])
    if misses:raise ValueError('人物分割缺失可见肢体，请换背景干净的照片；未把缺失的手脚直接送入建模。')
    target=Path(request['output']);image.crop(bounds).save(target)
    report['foreground']=str(target);Path(request['report']).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'ok':True,'foreground':str(target),'backend':'BiRefNet 1024 + '+('illustrated character foreground' if report.get('kind')=='illustrated_character' else 'MediaPipe person mask' if report.get('pose_mask') else 'portrait foreground'),'human':report,'size':list(image.crop(bounds).size)}

if __name__=='__main__':
    start_watchdog()
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['human','human_foreground']);parser.add_argument('request');parser.add_argument('response');args=parser.parse_args()
    try:
        request=json.loads(Path(args.request).read_text(encoding='utf-8'));result=inspect_person(request) if args.operation=='human' else foreground(request)
    except Exception as exc:
        import traceback;traceback.print_exc();result={'ok':False,'error':str(exc)}
    Path(args.response).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');sys.exit(0 if result['ok'] else 1)
