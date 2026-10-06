"""Conservative cleanup of neural meshes; preserve silhouette and color attributes."""
import sys
import math
import json
from pathlib import Path
import bpy

folder = Path(sys.argv[sys.argv.index('--') + 1])
description={}
if (folder/'recognition.json').exists():
    description=json.loads((folder/'recognition.json').read_text(encoding='utf-8')).get('description',{})
appearance=str(description.get('material_appearance','')).lower()
polished=any(word in appearance for word in ['polished','gloss','抛光','光泽'])
selected=json.loads((folder/'material.json').read_text(encoding='utf-8')) if (folder/'material.json').exists() else None
preserve_pbr=(folder/'pbr.json').exists()
bpy.ops.wm.read_factory_settings(use_empty=True)
if (folder / 'model.glb').exists():
    bpy.ops.import_scene.gltf(filepath=str(folder / 'model.glb'))
else:
    bpy.ops.wm.obj_import(filepath=str(folder / 'model.obj'))
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    raise RuntimeError('No reconstructed mesh')
if (folder/'ornament.json').exists():
    import importlib.util
    spec=importlib.util.spec_from_file_location('memorymade_blender_ornament',Path(__file__).with_name('blender_ornament.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.engrave(folder,meshes)
for obj in meshes:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    if hasattr(obj.data,'set_sharp_from_angle'):obj.data.set_sharp_from_angle(angle=math.radians(35))
    # No subdivision: it increases polygons but cannot recover missing photo detail.
    # No component deletion or remeshing: both can destroy handles/legs and colors.
    primary=not (selected and selected.get('preserve_component_materials')) or obj.name.startswith(('Seat','Backrest','Armrest'))
    if selected and not preserve_pbr and not selected.get('preserve_photo_color') and primary and not obj.name.startswith('PhotoDecal'):
        material=bpy.data.materials.new(selected['name']);material.use_nodes=True
        shader=material.node_tree.nodes.get('Principled BSDF')
        color=selected['color'].lstrip('#')
        shader.inputs['Base Color'].default_value=tuple((int(color[i:i+2],16)/255)**2.2 for i in (0,2,4))+(1,)
        shader.inputs['Roughness'].default_value=selected['roughness']
        shader.inputs['Metallic'].default_value=selected['metalness']
        if selected.get('key')=='碳纤维':
            import numpy as np
            nodes=material.node_tree.nodes;links=material.node_tree.links
            if not obj.data.uv_layers:
                bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.uv.smart_project(island_margin=.02);bpy.ops.object.mode_set(mode='OBJECT')
            coords=nodes.new('ShaderNodeTexCoord');mapping=nodes.new('ShaderNodeMapping');mapping.inputs['Scale'].default_value=(8,8,8)
            links.new(coords.outputs['UV'],mapping.inputs['Vector'])
            y,x=np.mgrid[0:256,0:256];u=x/32;v=y/32;horizontal=((np.floor(u)+np.floor(v))%4<2)
            height=.35+.25*np.where(horizontal,np.sin(v*math.tau),np.sin(u*math.tau))+.04*np.sin((x+y)*math.pi/2)
            def tile(name,pixels,colorspace):
                image=bpy.data.images.get(name)
                if image is None:
                    image=bpy.data.images.new(name,width=256,height=256,alpha=True);image.colorspace_settings.name=colorspace;image.pixels.foreach_set(pixels.astype('float32').reshape(-1));image.filepath_raw=str(folder/name);image.file_format='PNG';image.save();image.pack()
                texture=nodes.new('ShaderNodeTexImage');texture.image=image;links.new(mapping.outputs['Vector'],texture.inputs['Vector']);return texture
            rgb=np.stack([height*.13,height*.14,height*.16,np.ones_like(height)],-1)
            color_texture=tile('carbon-weave.png',rgb,'sRGB');links.new(color_texture.outputs['Color'],shader.inputs['Base Color'])
            gx=np.roll(height,-1,1)-np.roll(height,1,1);gy=np.roll(height,-1,0)-np.roll(height,1,0)
            normals=np.stack([-gx,-gy,np.ones_like(gx)],-1);normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
            normal_texture=tile('carbon-normal.png',np.concatenate([normals*.5+.5,np.ones((*height.shape,1))],-1),'Non-Color')
            normal=nodes.new('ShaderNodeNormalMap');normal.inputs['Strength'].default_value=.35;links.new(normal_texture.outputs['Color'],normal.inputs['Color']);links.new(normal.outputs['Normal'],shader.inputs['Normal']);shader.inputs['Coat Weight'].default_value=.22
        obj.data.materials.clear();obj.data.materials.append(material)
    elif obj.data.color_attributes:
        attribute = obj.data.color_attributes.active_color or obj.data.color_attributes[0]
        material = bpy.data.materials.new(obj.name + '_PhotoColor')
        material.use_nodes = True
        shader = material.node_tree.nodes.get('Principled BSDF')
        vertex = material.node_tree.nodes.new('ShaderNodeVertexColor')
        vertex.layer_name = attribute.name
        material.node_tree.links.new(vertex.outputs['Color'], shader.inputs['Base Color'])
        shader.inputs['Roughness'].default_value = selected['roughness'] if selected else (.4 if polished else .6)
        if selected:shader.inputs['Metallic'].default_value=selected['metalness']
        shader.inputs['Coat Weight'].default_value = .08 if polished else 0
        obj.data.materials.clear()
        obj.data.materials.append(material)
bpy.ops.object.select_all(action='DESELECT')
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.export_scene.gltf(filepath=str(folder / 'model.glb'), export_format='GLB', use_selection=True)
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'
for obj in meshes:
    if obj.name.startswith('PhotoDecal'):obj.select_set(False)
bpy.ops.wm.stl_export(filepath=str(folder / 'model.stl'), export_selected_objects=True, global_scale=1000)
for obj in meshes:obj.select_set(True)
bpy.ops.wm.obj_export(filepath=str(folder / 'model.obj'), export_selected_objects=True, global_scale=1000)
bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'model.blend'))
# A real render, not the uploaded photo, is used as the result preview.
from mathutils import Vector
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 40
scene.cycles.use_denoising = True
scene.view_settings.view_transform = 'AgX'
scene.render.resolution_x = 900
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.world = bpy.data.worlds.new('Studio')
scene.world.use_nodes = True
world_nodes=scene.world.node_tree.nodes
lighting=world_nodes.get('Background')
lighting.inputs[0].default_value = (.35,.35,.35,1)
lighting.inputs[1].default_value = .8
camera_background=world_nodes.new('ShaderNodeBackground')
camera_background.inputs[0].default_value=(.12,.12,.12,1)
path_node=world_nodes.new('ShaderNodeLightPath')
mix=world_nodes.new('ShaderNodeMixShader')
scene.world.node_tree.links.new(path_node.outputs['Is Camera Ray'],mix.inputs[0])
scene.world.node_tree.links.new(lighting.outputs[0],mix.inputs[1])
scene.world.node_tree.links.new(camera_background.outputs[0],mix.inputs[2])
scene.world.node_tree.links.new(mix.outputs[0],world_nodes.get('World Output').inputs['Surface'])
points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
center = sum(points, Vector()) / len(points)
span = max((point-center).length for point in points) * 2
camera_data = bpy.data.cameras.new('InspectionCamera')
camera = bpy.data.objects.new('InspectionCamera', camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.type = 'ORTHO'
camera_data.ortho_scale = span * 1.2
camera_data.clip_end = span * 100
camera_data.clip_start = max(span * .001, .000001)
light_data = bpy.data.lights.new('Key', 'AREA')
light = bpy.data.objects.new('Key', light_data)
scene.collection.objects.link(light)
light.location = center + Vector((span, -span, span*2))
light.rotation_euler = (center-light.location).to_track_quat('-Z', 'Y').to_euler()
light_data.energy = span*span*8
light_data.shape = 'DISK'
light_data.size = span*2
for name,position,power,size in [('Fill',(-1.5,-.2,.7),3.2,2),('Rim',(.2,1.5,1.4),7,1.2)]:
    data=bpy.data.lights.new(name,'AREA');obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
    obj.location=center+Vector(position)*span;obj.rotation_euler=(center-obj.location).to_track_quat('-Z','Y').to_euler();data.energy=span*span*power;data.size=span*size
for index, angle in enumerate([0, 120, 240]):
    radians = math.radians(angle)
    camera.location = center + Vector((math.sin(radians)*span*2, -math.cos(radians)*span*2, span*.8))
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(folder / f'view-{index}.png')
    bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'model.blend'))
print('MEMORYMADE_AI_BLENDER_OK')
