"""Isolated SigLIP2 material inference and VGGT visible-surface scene reconstruction."""
import argparse,json,os,sys
from pathlib import Path
BASE=Path(os.getenv('MEMORYMADE_RESOURCE_ROOT',str(Path(__file__).resolve().parents[2])))
from storage import ModelRoot
MODELS=ModelRoot(BASE)
os.environ['HF_HUB_OFFLINE']='1'

def material(request):
    import torch,numpy as np
    from PIL import Image,ImageOps
    from transformers import AutoModel,AutoProcessor
    from materials_ai import CATALOG
    image=ImageOps.exif_transpose(Image.open(request['image'])).convert('RGB')
    box=request.get('box')
    if box:
        x0,y0,x1,y1=box
        if x1>x0 and y1>y0:image=image.crop((max(0,x0),max(0,y0),min(image.width,x1),min(image.height,y1)))
    images=[image]
    w,h=image.size
    if min(w,h)>128:
        images += [image.crop((w*.2,h*.2,w*.8,h*.8)),image.crop((w*.3,h*.35,w*.7,h*.75))]
    prompts=[f'A close-up photograph of {row[1]}.' for row in CATALOG]
    path=MODELS/'siglip2-material'
    if not (path/'model.safetensors').exists():raise ValueError('材料识别模型尚未安装完成。')
    processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)
    model=AutoModel.from_pretrained(str(path),local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
    inputs=processor(text=prompts,images=images,padding='max_length',truncation=True,
                     max_length=model.config.text_config.max_position_embeddings,return_tensors='pt')
    inputs={k:v.to('cuda',dtype=torch.float16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}
    with torch.inference_mode():logits=model(**inputs).logits_per_image.float().mean(0)
    scores=torch.softmax(logits,dim=0).cpu().numpy();order=np.argsort(scores)[::-1][:5]
    ranking=[{'material':CATALOG[int(i)][0],'relative_match':round(float(scores[i]),4)} for i in order]
    visual=request.get('description',{}).get('material_appearance','')
    matches=[label for label in ['碳纤维','凯夫拉纤维','玻璃纤维','皮革','硅胶','橡胶','混凝土','木头','陶土'] if label in str(visual)]
    if 'carbon fiber' in str(visual).lower() or 'carbon fibre' in str(visual).lower():matches.append('碳纤维')
    best=ranking[0]['material'];margin=float(scores[order[0]]-scores[order[1]])
    agreement=best in matches
    # A weak visual match must remain a candidate, not an asserted composition.
    selected=best if scores[order[0]]>=.18 and margin>=.035 else '未知表面'
    return {'ok':True,'material':selected,'candidates':ranking,'qwen_appearance':visual,'cross_agreement':agreement,
            'backend':'Google SigLIP2 base + Qwen appearance cross-check','verified_composition':False,
            'note':'按照片纹理估计材质外观；分数仅为候选相对匹配度，不证明真实成分。'}

def scene(request):
    import torch,numpy as np,trimesh
    from safetensors.torch import load_file
    source=json.loads((MODELS/'vggt/source-code.json').read_text(encoding='utf-8'))
    sys.path.insert(0,source['folder'])
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    paths=request['images']
    if not 1<=len(paths)<=4:raise ValueError('本机场景重建一次使用 1–4 张照片。')
    model=VGGT(enable_track=False,enable_depth=False).to(device='cuda',dtype=torch.bfloat16).eval()
    # Upstream runs these prediction heads outside autocast on float32 tokens.
    model.camera_head.float();model.point_head.float()
    weights=load_file(str(MODELS/'vggt/model.safetensors'),device='cpu')
    weights={name:value for name,value in weights.items() if not name.startswith(('track_head.','depth_head.'))}
    model.load_state_dict(weights,strict=True);del weights
    images=load_and_preprocess_images(paths).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(images)
    points=prediction['world_points'][0].float().cpu().numpy()
    confidence=prediction['world_points_conf'][0].float().cpu().numpy()
    colors=images.permute(0,2,3,1).float().cpu().numpy()
    meshes=[]
    for index,grid in enumerate(points):
        h,w=grid.shape[:2];valid=np.isfinite(grid).all(2)&(confidence[index]>max(1.0,float(np.percentile(confidence[index],20))))
        ids=np.arange(h*w).reshape(h,w)
        faces=np.stack([np.stack([ids[:-1,:-1],ids[1:,:-1],ids[:-1,1:]],-1),np.stack([ids[:-1,1:],ids[1:,:-1],ids[1:,1:]],-1)],-2).reshape(-1,3)
        flat=grid.reshape(-1,3);triangles=flat[faces]
        lengths=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2)
        local_scale=np.linalg.norm(grid[1:,:-1]-grid[:-1,:-1],axis=2)
        threshold=max(float(np.median(local_scale[np.isfinite(local_scale)]))*12,1e-5)
        keep=valid.reshape(-1)[faces].all(1)&(lengths.max(1)<threshold)
        mesh=trimesh.Trimesh(vertices=flat,faces=faces[keep],vertex_colors=(colors[index].reshape(-1,3)*255).clip(0,255).astype('uint8'),process=False)
        mesh.remove_unreferenced_vertices();meshes.append(mesh)
    mesh=trimesh.util.concatenate(meshes)
    if len(mesh.faces)<100:raise ValueError('场景表面有效区域不足，请上传更清晰、相互重叠的照片。')
    # VGGT coordinate scale is not a measured physical dimension.
    mesh.apply_scale(float(request.get('width_mm',120))/float(mesh.extents.max()));mesh.apply_translation(-mesh.bounds.mean(0))
    folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    glb=mesh.copy();glb.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0]));glb.apply_scale(.001);glb.export(folder/'model.glb')
    printable=mesh.copy();printable.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]));printable.export(folder/'model.stl');printable.export(folder/'model.obj')
    np.savez_compressed(folder/'scene-cameras.npz',pose_enc=prediction['pose_enc'].float().cpu().numpy())
    return {'ok':True,'backend':'VGGT visible scene','glb':str(folder/'model.glb'),'stl':str(folder/'model.stl'),'obj':str(folder/'model.obj'),
            'faces':len(mesh.faces),'views':len(paths),'peak_cuda_gib':round(torch.cuda.max_memory_allocated()/1024**3,2),
            'watertight':False,'scale':'user requested display size, not measured real-world scale',
            'note':'照片可见表面与相机关系估计；不补造不可见背面，不等同完整封闭建筑模型。'}

if __name__=='__main__':
    from disk_guard import start_watchdog
    start_watchdog()
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['material','scene']);parser.add_argument('request');parser.add_argument('response');args=parser.parse_args()
    try:result=(material if args.operation=='material' else scene)(json.loads(Path(args.request).read_text(encoding='utf-8')))
    except Exception as exc:
        import traceback;traceback.print_exc();result={'ok':False,'error':str(exc)}
    Path(args.response).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');sys.exit(0 if result['ok'] else 1)
