import math
import os
import sys

import bpy

argv = sys.argv
args = argv[argv.index('--') + 1:] if '--' in argv else []
if len(args) < 5:
    raise SystemExit('usage: blender_refine.py input.glb output.glb output.stl output.obj output.blend')

input_glb, output_glb, output_stl, output_obj, output_blend = args[:5]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_glb)
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    raise SystemExit('no mesh imported')

bpy.ops.object.select_all(action='DESELECT')
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
obj.name = 'MEMORYMADE_Object'

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

try:
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(32))
except Exception:
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass

size = max(max(obj.dimensions), 1.0)
try:
    bevel = obj.modifiers.new(name='MEMORYMADE_Edge', type='BEVEL')
    bevel.width = min(max(size * 0.004, 0.15), 0.8)
    bevel.segments = 2
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = math.radians(28)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bevel.name)
except Exception:
    pass

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB', use_selection=True)
bpy.ops.wm.stl_export(filepath=output_stl, export_selected_objects=True)
bpy.ops.wm.obj_export(filepath=output_obj, export_selected_objects=True)
bpy.ops.wm.save_as_mainfile(filepath=output_blend)
print('MEMORYMADE_BLENDER_OK')
