"""A solid engraved memory coin, with an optional image decal."""
import sys, json, math
from pathlib import Path
import bpy
folder=Path(sys.argv[sys.argv.index('--')+1]);settings=json.loads((folder/'coin.json').read_text(encoding='utf-8'))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cylinder_add(vertices=192,radius=.02,depth=.004)
coin=bpy.context.object;coin.name='MemoryCoin'
bevel=coin.modifiers.new('Soft rim','BEVEL');bevel.width=.00035;bevel.segments=4
bpy.ops.object.modifier_apply(modifier=bevel.name)
text=settings.get('engraving','').strip()
if text:
    bpy.ops.object.text_add(location=(0,-.014 if settings.get('photo') else 0,.002))
    letters=bpy.context.object;letters.data.body=text[:18];letters.data.align_x='CENTER';letters.data.align_y='CENTER';letters.data.size=.0035;letters.data.extrude=.0004
    for name in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/arial.ttf']:
        if Path(name).exists():letters.data.font=bpy.data.fonts.load(name);break
    bpy.context.view_layer.update()
    available_width=.023 if settings.get('photo') else .031
    if letters.dimensions.x>available_width:letters.scale.x=available_width/letters.dimensions.x
    bpy.ops.object.convert(target='MESH')
    bpy.context.view_layer.objects.active=coin
    modifier=coin.modifiers.new('Engraved memory','BOOLEAN');modifier.operation='DIFFERENCE';modifier.solver='EXACT';modifier.object=letters
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(letters,do_unlink=True)
    import bmesh
    bm=bmesh.new();bm.from_mesh(coin.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-8)
    bmesh.ops.dissolve_limit(bm,angle_limit=.001,use_dissolve_boundaries=True,verts=list(bm.verts),edges=list(bm.edges))
    bmesh.ops.triangulate(bm,faces=list(bm.faces),quad_method='BEAUTY',ngon_method='BEAUTY')
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(coin.data);bm.free();coin.data.update()
if settings.get('photo'):
    radius=.013
    vertices=[(0,.002,.00203)]+[(radius*math.cos(i*2*math.pi/128),.002+radius*math.sin(i*2*math.pi/128),.00203) for i in range(128)]
    faces=[(0,i+1,(i+1)%128+1) for i in range(128)]
    mesh=bpy.data.meshes.new('PhotoDecalMesh');mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new('PhotoDecal',mesh);bpy.context.collection.objects.link(obj)
    uv=mesh.uv_layers.new()
    for poly in mesh.polygons:
        for loop in poly.loop_indices:
            vertex=mesh.vertices[mesh.loops[loop].vertex_index].co
            uv.data[loop].uv=(vertex.x/(2*radius)+.5,(vertex.y-.002)/(2*radius)+.5)
    material=bpy.data.materials.new('MemoryPhoto');material.use_nodes=True
    texture=material.node_tree.nodes.new('ShaderNodeTexImage');texture.image=bpy.data.images.load(settings['photo']);texture.image.pack()
    material.node_tree.links.new(texture.outputs['Color'],material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    material.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.45
    mesh.materials.append(material)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(filepath=str(folder/'model.glb'),export_format='GLB')
exec(compile(Path(__file__).with_name('blender_ai_refine.py').read_text(encoding='utf-8'),str(Path(__file__).with_name('blender_ai_refine.py')),'exec'))
