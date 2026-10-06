"""Local, pinned foreground extraction and diffusion shape reconstruction."""
import sys, json, argparse, os
from pathlib import Path
BASE=Path(os.getenv('MEMORYMADE_RESOURCE_ROOT',str(Path(__file__).resolve().parents[2])))
from storage import ModelRoot
MODELS=ModelRoot(BASE)
HY21_ROOT=BASE if (BASE/'tools/Hunyuan3D-2.1/hy3dshape/hy3dshape/pipelines.py').exists() else Path('D:/1223456789')
HY21_TOOLS=HY21_ROOT/'tools/Hunyuan3D-2.1'
HY21_MODELS=HY21_ROOT/'assets/models/hunyuan3d-2.1'
os.environ.setdefault('HF_HOME',str(HY21_ROOT/'cache/huggingface'))
os.environ['HF_HUB_OFFLINE']='1'

def segment(request):
    import torch, numpy as np
    from PIL import Image, ImageOps
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    image=ImageOps.exif_transpose(Image.open(request['image'])).convert('RGBA')
    # Respect an intentionally supplied alpha mask.
    if image.getextrema()[3][0] < 255:
        backend='uploaded alpha'
    else:
        model=AutoModelForImageSegmentation.from_pretrained(str(MODELS/'birefnet-lite'),trust_remote_code=True,local_files_only=True).eval().to('cuda')
        transform=transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        tensor=transform(image.convert('RGB')).unsqueeze(0).to('cuda')
        with torch.inference_mode():
            prediction=model(tensor)[-1].sigmoid()[0,0].float().cpu().numpy()
        mask=Image.fromarray((prediction*255).clip(0,255).astype('uint8')).resize(image.size,Image.Resampling.LANCZOS)
        image.putalpha(mask)
        backend='BiRefNet Lite 1024'
    box=request.get('box')
    if box:
        x1,y1,x2,y2=box
        margin=max(x2-x1,y2-y1)*.10
        bounds=(max(0,int(x1-margin)),max(0,int(y1-margin)),min(image.width,int(x2+margin)),min(image.height,int(y2+margin)))
        if bounds[2]>bounds[0] and bounds[3]>bounds[1]:image=image.crop(bounds)
    alpha=np.asarray(image)[:,:,3]
    count=int((alpha>128).sum())
    if count<100:raise ValueError('BiRefNet 没有找到有效主体，请改用背景简单的照片。')
    path=Path(request['output']);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
    from scipy.ndimage import binary_fill_holes,label
    binary=alpha>128
    holes=binary_fill_holes(binary)&~binary
    return {'ok':True,'foreground':str(path),'backend':backend,'size':list(image.size),'foreground_pixels':count,'hole_pixels':int(holes.sum()),'holes':int(label(holes)[1])}

def refine_human_shape(mesh,request,folder):
    if request.get('subject_kind')!='human':return mesh,{}
    from human_mesh import clean_human_mesh,check_human_silhouette
    mesh.export(folder/'raw-neural.glb')
    recognition_path=folder/'recognition.json'
    description=json.loads(recognition_path.read_text(encoding='utf-8')).get('description',{}) if recognition_path.is_file() else {}
    # Shape alone cannot distinguish a curved cord from compact floating noise.
    # When observed props/cords are present, every detached component needs review.
    protect_accessories=any(isinstance(description.get(key),list) and bool(description[key]) for key in ['thin_structures','held_objects'])
    cleaned,topology=clean_human_mesh(mesh,folder,protect_accessories=protect_accessories)
    human=json.loads(Path(request['human_report']).read_text(encoding='utf-8'))
    silhouette=check_human_silhouette(cleaned,request['foreground'],human,folder)
    review={'topology':topology,'silhouette':silhouette,'body_kind':human['kind'],'warnings':human['warnings']}
    (folder/'human-quality.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    return cleaned,review

def hunyuan(request):
    import torch,numpy as np,trimesh
    from PIL import Image,ImageOps
    sources=json.loads((MODELS/'upgrade-sources.json').read_text())
    sys.path.insert(0,str(BASE/'tools'/('Hunyuan3D-2-'+sources['hunyuan_source']['commit'])))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    ImageOps.exif_transpose(Image.open(request['image'])).save(folder/'source.png')
    image=Image.open(request['foreground']).convert('RGBA');image.save(folder/'foreground.png')
    model_path=MODELS/'hunyuan3d-mini/hunyuan3d-dit-v2-mini'
    multiview=bool(request.get('views'))
    if multiview:
        candidates=[Path(os.getenv('MEMORYMADE_WRITE_ROOT',str(BASE)))/'assets/models/hunyuan3d-multiview/hunyuan3d-dit-v2-mv',Path('D:/1223456789/assets/models/hunyuan3d-multiview/hunyuan3d-dit-v2-mv'),MODELS/'hunyuan3d-multiview/hunyuan3d-dit-v2-mv']
        model_path=next((p for p in candidates if (p/'model.fp16.safetensors').exists()),None)
        if model_path is None:raise ValueError('多视角模型尚未完成安装；没有把多张照片冒充融合重建。')
        image={tag:Image.open(path).convert('RGBA') for tag,path in request['views'].items()}
    pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_single_file(str(model_path/'model.fp16.safetensors'),str(model_path/'config.yaml'),device='cpu',dtype=torch.float16,use_safetensors=True)
    # The pinned upstream offload helper expects DiffusionPipeline.components,
    # although this shape pipeline does not inherit DiffusionPipeline.
    pipeline.components={'conditioner':pipeline.conditioner,'model':pipeline.model,'vae':pipeline.vae}
    pipeline.enable_model_cpu_offload(device='cuda')
    if not multiview:
        processed=pipeline.image_processor(image,to_tensor=False)
        Image.fromarray(processed['image']).save(folder/'input.png')
    else:
        canvas=Image.new('RGBA',(512*len(image),512),'white')
        for index,value in enumerate(image.values()):
            canvas.paste(ImageOps.contain(value,(512,512)),(index*512,0))
        canvas.convert('RGB').save(folder/'input.png')
    # This source consults device during latent creation even when modules are offloaded.
    pipeline.device=torch.device('cuda')
    steps=int(request.get('steps',30));resolution=int(request.get('resolution',256))
    torch.cuda.reset_peak_memory_stats()
    def generate(seed):
        candidate=pipeline(image=image,num_inference_steps=steps,guidance_scale=5.0,octree_resolution=resolution,num_chunks=int(request.get('num_chunks',4000)),mc_algo='mc',generator=torch.Generator(device='cuda').manual_seed(seed))[0]
        if candidate is None or len(candidate.faces)<10:raise ValueError('Hunyuan3D 未生成有效表面')
        return candidate
    seed=20261005
    if request.get('subject_kind')=='human':
        from human_mesh import retry_human_shape
        mesh,human_review,seed=retry_human_shape(generate,lambda mesh:refine_human_shape(mesh,request,folder),folder)
    else:mesh,human_review=refine_human_shape(generate(seed),request,folder)
    mesh.apply_scale(float(request.get('width_mm',120))/float(mesh.extents.max()))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    # Hunyuan outputs Y-up. glTF uses meters; printing formats below use Z-up/mm.
    color=request.get('color','#9B663F').lstrip('#')
    rgb=[int(color[i:i+2],16) for i in (0,2,4)]
    mesh.visual.vertex_colors=np.tile(rgb+[255],(len(mesh.vertices),1))
    glb=mesh.copy();glb.apply_scale(.001);glb.export(folder/'model.glb')
    printable=mesh.copy();printable.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    printable.export(folder/'model.stl');printable.export(folder/'model.obj')
    return {'ok':True,'backend':'Hunyuan3D-2 multiview' if multiview else 'Hunyuan3D-2 mini','view_tags':list(image) if multiview else ['front'],'device':'cuda / CPU offload','glb':str(folder/'model.glb'),'stl':str(folder/'model.stl'),'obj':str(folder/'model.obj'),'faces':len(mesh.faces),'vertices':len(mesh.vertices),'steps':steps,'peak_cuda_gib':round(torch.cuda.max_memory_allocated()/1024**3,2),'color_source':'user selected uniform color; texture not reconstructed','seed':seed,'human_review':human_review}

def _load_hunyuan21_pipeline():
    import torch
    sys.path.insert(0, str(HY21_TOOLS/'hy3dshape'))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    model_dir = HY21_MODELS/'hunyuan3d-dit-v2-1'
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(model_dir/'model.fp16.ckpt'), str(model_dir/'config.yaml'),
        device='cpu', dtype=torch.float16, use_safetensors=False)
    pipe.components={'conditioner':pipe.conditioner,'model':pipe.model,'vae':pipe.vae}
    pipe.enable_model_cpu_offload(device='cuda')
    pipe.device=torch.device('cuda')
    return pipe

def hunyuan21(request):
    import torch,numpy as np,trimesh
    from PIL import Image,ImageOps
    folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    ImageOps.exif_transpose(Image.open(request['image'])).save(folder/'source.png')
    image=Image.open(request['foreground']).convert('RGBA');image.save(folder/'foreground.png')
    pipeline=_load_hunyuan21_pipeline()
    steps=int(request.get('steps',50));resolution=int(request.get('resolution',384))
    torch.cuda.reset_peak_memory_stats()
    kwargs=dict(image=image,num_inference_steps=steps,guidance_scale=5.0,octree_resolution=resolution,num_chunks=int(request.get('num_chunks',4000)))
    # Keep the requested quality; a resource error must not silently rerun at another quality.
    def generate(seed):
        candidate=pipeline(mc_algo=request.get('mc_algo','mc'),generator=torch.Generator(device='cuda').manual_seed(seed),**kwargs)[0]
        if candidate is None or len(candidate.faces)<10:raise ValueError('Hunyuan3D 2.1 未生成有效表面')
        return candidate
    seed=20261005
    if request.get('subject_kind')=='human':
        from human_mesh import retry_human_shape
        mesh,human_review,seed=retry_human_shape(generate,lambda mesh:refine_human_shape(mesh,request,folder),folder)
    else:mesh,human_review=refine_human_shape(generate(seed),request,folder)
    mesh.apply_scale(float(request.get('width_mm',120))/float(mesh.extents.max()))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    color=request.get('color','#9B663F').lstrip('#');rgb=[int(color[i:i+2],16) for i in (0,2,4)]
    mesh.visual.vertex_colors=np.tile(rgb+[255],(len(mesh.vertices),1))
    glb=mesh.copy();glb.apply_scale(.001);glb.export(folder/'model.glb')
    printable=mesh.copy();printable.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    printable.export(folder/'model.stl');printable.export(folder/'model.obj')
    return {'ok':True,'backend':'Hunyuan3D-2.1 Shape','glb':str(folder/'model.glb'),'stl':str(folder/'model.stl'),'obj':str(folder/'model.obj'),'faces':len(mesh.faces),'vertices':len(mesh.vertices),'steps':steps,'resolution':resolution,'peak_cuda_gib':round(torch.cuda.max_memory_allocated()/1024**3,2),'color_source':'user selected uniform color; texture not reconstructed','seed':seed,'human_review':human_review}

def hunyuan21_pbr(request):
    shape=hunyuan21(request)
    folder=Path(request['output_dir'])
    try:
        sys.path.insert(0,str(HY21_TOOLS))
        import torchvision_fix
        torchvision_fix.apply_fix()
        sys.path.insert(0,str(HY21_TOOLS/'hy3dpaint'))
        from textureGenPipeline import Hunyuan3DPaintConfig,Hunyuan3DPaintPipeline
    except Exception as exc:
        raise RuntimeError('Hunyuan3D 2.1 PBR 运行环境尚未编译完成：'+str(exc)) from exc
    import torch as _torch
    _vram=float(_torch.cuda.get_device_properties(0).total_memory)/1024**3
    if _vram < 16:
        cfg=Hunyuan3DPaintConfig(max_num_view=4,resolution=384)
        cfg.render_size=1024
        cfg.texture_size=2048
    else:
        cfg=Hunyuan3DPaintConfig(max_num_view=6,resolution=512)
    cfg.multiview_pretrained_path=str(HY21_MODELS)
    cfg.dino_ckpt_path=str(HY21_ROOT/'assets/models/dinov2-giant')
    cfg.realesrgan_ckpt_path=str(HY21_ROOT/'assets/models/realesrgan/RealESRGAN_x4plus.pth')
    cfg.raster_mode='cr';cfg.bake_mode='back_sample'
    pipe=Hunyuan3DPaintPipeline(cfg)
    textured=folder/'textured_mesh.obj'
    out=pipe(mesh_path=shape['obj'],image_path=request['image'],output_mesh_path=str(textured),use_remesh=True,save_glb=True)
    glb=Path(str(out).replace('.obj','.glb'))
    return {**shape,'backend':'Hunyuan3D-2.1 Shape + PBR','glb':str(glb),'stl':shape['stl'],'obj':str(out),'color_source':'PBR albedo + metallic/roughness','pbr':True}

if __name__=='__main__':
    from disk_guard import start_watchdog
    start_watchdog()
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['segment','hunyuan','hunyuan21','hunyuan21_pbr']);parser.add_argument('request');parser.add_argument('response');args=parser.parse_args()
    try:
        request=json.loads(Path(args.request).read_text(encoding='utf-8'))
        result=segment(request) if args.operation=='segment' else (hunyuan21_pbr(request) if args.operation=='hunyuan21_pbr' else (hunyuan21(request) if args.operation=='hunyuan21' else hunyuan(request)))
    except Exception as exc:
        import traceback;traceback.print_exc();result={'ok':False,'error':str(exc)}
    Path(args.response).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.exit(0 if result['ok'] else 1)



