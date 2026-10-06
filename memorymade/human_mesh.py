"""Human-specific topology checks; no synthetic limb bridges or global decimation."""
from pathlib import Path
import json
import numpy as np
import trimesh

class HumanQualityError(ValueError):
    pass


def retry_human_shape(generate, validate, folder, seeds=(20261005,20261006)):
    """Try another diffusion seed at the same resolution; retain rejected evidence."""
    try:
        from .disk_guard import ensure_disk_space
    except ImportError:
        from disk_guard import ensure_disk_space
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);attempts=[]
    def preserve_attempt(index):
        for name in ['raw-neural.glb','human-mesh-review.json','human-mesh-silhouette.png','human-silhouette-review.json','human-quality.json']:
            path=folder/name
            if path.is_file():path.replace(folder/f'human-attempt-{index}-{name}')
        (folder/'human-generation-attempts.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
    for index,seed in enumerate(seeds,1):
        ensure_disk_space('人物细部重建与连接复核')
        try:
            mesh=generate(seed)
        except Exception as exc:
            # Resource/runtime failures are recorded but never silently retried
            # at lower resolution or mistaken for a rejected topology check.
            attempts.append({'attempt':index,'seed':seed,'passed':False,'stage':'generation','retryable':False,'reason':str(exc),'error_type':type(exc).__name__})
            preserve_attempt(index)
            raise
        try:
            mesh,review=validate(mesh)
        except HumanQualityError as exc:
            attempts.append({'attempt':index,'seed':seed,'passed':False,'stage':'validation','reason':str(exc)})
            preserve_attempt(index)
            if index==len(seeds):
                raise HumanQualityError(f'人物以相同精细度生成并检查了 {index} 次，仍未通过连接或轮廓检查。'+str(exc)) from exc
            continue
        attempts.append({'attempt':index,'seed':seed,'passed':True})
        (folder/'human-generation-attempts.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
        return mesh,review,seed
    raise HumanQualityError('未配置人物生成尝试。')

def clean_human_mesh(mesh, folder=None, protect_accessories=False):
    mesh=mesh.copy()
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)<100:
        raise HumanQualityError('人物网格无效，请改用清晰的人物照片。')
    before=len(mesh.faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces)<100:
        raise HumanQualityError('人物网格去除无效面后缺少有效表面，请改用清晰的人物照片。')
    groups=trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(mesh.faces)),engine='scipy')
    areas=mesh.area_faces
    groups=sorted(groups,key=lambda g:float(areas[g].sum()),reverse=True)
    primary=groups[0];area=float(areas[primary].sum());span=float(np.linalg.norm(np.ptp(mesh.vertices[np.unique(mesh.faces[primary])],axis=0)))
    from scipy.spatial import cKDTree
    primary_tree=cKDTree(mesh.vertices[np.unique(mesh.faces[primary])]) if len(groups)>1 else None
    removed=[];ambiguous=[]
    for group in groups[1:]:
        points=mesh.vertices[np.unique(mesh.faces[group])]
        ratio=float(areas[group].sum()/max(area,1e-12));extent=float(np.linalg.norm(np.ptp(points,axis=0))/max(span,1e-12))
        eigenvalues=np.maximum(np.linalg.eigvalsh(np.cov(points,rowvar=False,bias=True)),0)
        elongation=float(np.sqrt(eigenvalues[-1]/max(eigenvalues[-2],1e-16)))
        gap=float(primary_tree.query(points,k=1)[0].min()/max(span,1e-12))
        thin=elongation>=4 and extent>=.004
        near=gap<=.008
        entry={'faces':int(len(group)),'area_ratio':round(ratio,6),'relative_extent':round(extent,5),
               'elongation':round(elongation,3),'relative_gap':round(gap,6),
               'suspected_thin_structure':thin,'near_attachment':near}
        # A thin rope or a fragment near an attachment is not disposable noise,
        # even if its surface area is very small relative to the human body.
        if ratio<=.003 and extent<=.06 and not thin and not near and not protect_accessories:removed.append(entry)
        else:ambiguous.append(entry)
    report={'input_faces':before,'components_before':len(groups),'removed_small_floaters':removed,'ambiguous_components':ambiguous,'accessory_protection':bool(protect_accessories),'surface_detail_preserved':True,'anatomy_verified':False,'semantic_attachments_verified':False,'synthetic_connections_added':False}
    if ambiguous:
        report['passed']=False
        if folder:Path(folder,'human-mesh-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise HumanQualityError('人物模型存在分离部件，可能包含断开的绳子、玩具、饰品或肢体；已暂停交付，不会将细长部件当垃圾删除或随意搭桥。请补拍手、玩具和绳子两端连接位置的清晰照片，并检查原始模型。')
    result=mesh.submesh([primary],append=True,repair=False)
    trimesh.repair.fix_normals(result,multibody=False)
    if not result.is_watertight:trimesh.repair.fill_holes(result)
    report.update(output_faces=len(result.faces),components_after=1,watertight=bool(result.is_watertight),passed=bool(result.is_watertight))
    if folder:Path(folder,'human-mesh-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result.is_watertight:
        raise HumanQualityError('人物表面存在无法保守修复的开口，已保留模型供复核；不会用大块封面掩盖缺失的人体结构。')
    return result,report

def check_human_silhouette(mesh, foreground, human_report, folder):
    from PIL import Image,ImageDraw
    from scipy.ndimage import distance_transform_edt
    size=256
    alpha=Image.open(foreground).convert('RGBA').getchannel('A');bbox=alpha.getbbox()
    source=np.asarray(alpha.crop(bbox).resize((size,size)))>80
    xy=mesh.vertices[:,[0,1]].copy();xy[:,1]*=-1
    low=xy.min(axis=0);extent=np.maximum(xy.max(axis=0)-low,1e-8)
    pixels=np.clip(np.round((xy-low)/extent*(size-1)),0,size-1).astype(int)
    image=Image.new('L',(size,size));draw=ImageDraw.Draw(image)
    for face in mesh.faces:draw.polygon([tuple(p) for p in pixels[face]],fill=255)
    predicted=np.asarray(image)>0
    overlap=int((predicted&source).sum());union=int((predicted|source).sum())
    iou=overlap/max(union,1)
    precision=overlap/max(int(predicted.sum()),1);recall=overlap/max(int(source.sum()),1)
    # Landmarks are expressed in the saved foreground canvas, not the original crop.
    distances=distance_transform_edt(~predicted)
    reliable=[]
    for landmark in human_report.get('foreground_landmarks',[]):
        if landmark['index'] not in [11,12,13,14,15,16,23,24,25,26,27,28] or landmark['visibility']<.75:continue
        x=(landmark['x']*alpha.width-bbox[0])/max(bbox[2]-bbox[0],1)*size
        y=(landmark['y']*alpha.height-bbox[1])/max(bbox[3]-bbox[1],1)*size
        if 0<=x<size and 0<=y<size:reliable.append(float(distances[int(y),int(x)])<=size*.08)
    support=sum(reliable)/len(reliable) if reliable else None
    result={'silhouette_iou':round(iou,4),'silhouette_precision':round(precision,4),'silhouette_recall':round(recall,4),'visible_joint_support_ratio':round(support,4) if support is not None else None,'visible_joints_checked':len(reliable),'passed':iou>=.55 and precision>=.70 and recall>=.70 and (support is None or support>=.85),'anatomy_verified':False}
    image.save(Path(folder,'human-mesh-silhouette.png'))
    Path(folder,'human-silhouette-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['passed']:raise HumanQualityError('人物三维轮廓或可见肢体与照片相差较大，已暂停交付并保留复核图。请补充更清晰的全身正面照片或不同角度。')
    return result
