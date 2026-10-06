"""Surface-projected, closed polygon cutters for genuine traditional engraving."""
import json
import bpy
from mathutils import Vector

def engrave(folder,meshes):
    config=folder/'ornament.json'
    if not config.exists():return
    settings=json.loads(config.read_text(encoding='utf-8'))
    if not settings.get('added_by_user'):return
    geometry=json.loads((folder/'ornament-cutter.json').read_text(encoding='utf-8'))
    target=max((obj for obj in meshes if not obj.name.startswith('PhotoDecal')),key=lambda obj:len(obj.data.polygons))
    bpy.context.view_layer.update()
    world=[target.matrix_world@Vector(p) for p in target.bound_box]
    low=Vector(tuple(min(p[i] for p in world) for i in range(3)));high=Vector(tuple(max(p[i] for p in world) for i in range(3)))
    span=high-low;center=(high+low)/2;coin=settings.get('kind')=='coin'
    depth=float(settings['depth_mm'])/1000
    raised=settings.get('style')=='浮雕（凸起）'
    direction=Vector((0,0,-1) if coin else (0,1,0))
    width=span.x*(.52 if coin else .65);height=(span.y if coin else span.z)*(.25 if coin else .23)
    center.z=low.z+span.z*.60 if not coin else high.z
    if coin:center.y=low.y+span.y*.20
    inverse=target.matrix_world.inverted();raydirection=(inverse.to_3x3()@direction).normalized()
    cutters=[];hits=0
    for piece in geometry:
        vertices=[];valid=True;cache={}
        for x,y,t in piece['vertices']:
            key=(x,y)
            if key not in cache:
                origin=Vector((center.x+x*width/2,center.y+y*height/2,high.z+span.z+depth)) if coin else Vector((center.x+x*width/2,low.y-span.y-depth,center.z+y*height/2))
                found,point,normal,_=target.ray_cast(inverse@origin,raydirection)
                if not found:valid=False;break
                point=target.matrix_world@point;normal=(target.matrix_world.to_3x3().inverted().transposed()@normal).normalized()
                if abs(normal.dot(direction))<.45:valid=False;break
                cache[key]=point;hits+=1
            vertices.append(cache[key]+direction*((t*1.5-(1 if raised else .5))*depth))
        if not valid:continue
        faces=[list(reversed(face)) for face in piece['faces']]
        mesh=bpy.data.meshes.new('ClosedTraditionalMotif');mesh.from_pydata(vertices,[],faces);mesh.update()
        cutter=bpy.data.objects.new('AddedTraditionalMotif',mesh);bpy.context.collection.objects.link(cutter);cutters.append(cutter)
    if not cutters:raise ValueError('正面没有适合连续浅雕的实体表面，请保留照片雕刻或选择其他模型。')
    bpy.context.view_layer.update()
    for cutter in cutters:
        bpy.context.view_layer.objects.active=target
        modifier=target.modifiers.new('ActualTraditionalEngraving','BOOLEAN');modifier.operation='UNION' if raised else 'DIFFERENCE';modifier.solver='MANIFOLD';modifier.object=cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name);bpy.data.objects.remove(cutter,do_unlink=True)
        if not len(target.data.polygons):raise ValueError('浅雕切割未生成有效表面，原始模型保留。')
    (folder/'ornament-report.json').write_text(json.dumps({'pattern':settings['pattern'],'depth_mm':settings['depth_mm'],'style':settings.get('style','阴刻（凹槽）'),'added_by_user':True,'projected_samples':hits,'cutters':len(cutters),'method':'Shapely closed outlines + Earcut triangulation + Blender projected Boolean','note':'用户新增的正面装饰，不是识别到的原照片纹样；缺少连续实体表面的纹样分段会跳过。'},ensure_ascii=False,indent=2),encoding='utf-8')
