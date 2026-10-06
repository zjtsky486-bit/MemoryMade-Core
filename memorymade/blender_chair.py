"""Explicit mechanical prior: open backrest slots, lift, star base and casters."""
import bpy,json,math,sys
from pathlib import Path
from mathutils import Vector
folder=Path(sys.argv[sys.argv.index('--')+1]);settings=json.loads((folder/'chair.json').read_text(encoding='utf-8'))
bpy.ops.wm.read_factory_settings(use_empty=True)
parts=[]
def surface(name,color,roughness=.5,metal=0):
 m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=roughness;p.inputs['Metallic'].default_value=metal;return m
upholstery=surface('Chair upholstery',(.05,.055,.065),.6)
metal=surface('Chair mechanism',(.15,.16,.17),.25,.8)
rubber=surface('Caster rubber',(.018,.019,.02),.82)
accent=surface('Accent stitching',(.35,.035,.03),.5)
def finish(obj,name,material,bevel=0):
 obj.name=name;obj.data.materials.append(material)
 if bevel:
  modifier=obj.modifiers.new('Rounded edges','BEVEL');modifier.width=bevel;modifier.segments=4;bpy.context.view_layer.objects.active=obj;bpy.ops.object.modifier_apply(modifier=modifier.name)
 for polygon in obj.data.polygons:polygon.use_smooth=True
 parts.append(obj);return obj
def box(name,center,size,material,bevel=.0007):
 bpy.ops.mesh.primitive_cube_add(size=1,location=center);obj=bpy.context.object;obj.dimensions=size;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);return finish(obj,name,material,bevel)
def rod(name,start,end,radius,material):
 start=Vector(start);end=Vector(end);bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=radius,depth=(end-start).length,location=(start+end)/2);obj=bpy.context.object;obj.rotation_euler=(end-start).to_track_quat('Z','Y').to_euler();return finish(obj,name,material,.00025)
seat=box('Seat',(0,-.004,.052),(.045,.043,.009),upholstery,.0025)
# A shaped, closed plate, so subtractive openings remain real through-holes.
profile=[(-.017,.058),(-.022,.067),(-.021,.093),(-.025,.104),(-.020,.116),(-.012,.122),(.012,.122),(.020,.116),(.025,.104),(.021,.093),(.022,.067),(.017,.058)]
vertices=[(x,.016+y,z) for y in [-.004,.004] for x,z in profile];n=len(profile)
faces=[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
mesh=bpy.data.meshes.new('BackrestMesh');mesh.from_pydata(vertices,[],faces);mesh.update();back=bpy.data.objects.new('Backrest',mesh);bpy.context.collection.objects.link(back);bpy.context.view_layer.objects.active=back;back.select_set(True)
finish(back,'Backrest',upholstery,.0012)
for index in range(int(settings['holes'])):
 x=(index-(settings['holes']-1)/2)*.016
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=.0043,depth=.035,location=(x,.016,.108),rotation=(math.pi/2,0,0))
 cutter=bpy.context.object;cutter.scale=(1,.62,1);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
 modifier=back.modifiers.new(f'Actual through slot {index+1}','BOOLEAN');modifier.operation='DIFFERENCE';modifier.solver='MANIFOLD';modifier.object=cutter
 bpy.context.view_layer.objects.active=back;bpy.ops.object.modifier_apply(modifier=modifier.name);bpy.data.objects.remove(cutter,do_unlink=True)
rod('Gas lift',(0,0,.016),(0,0,.047),.003,metal)
rod('Lift sleeve',(0,0,.016),(0,0,.034),.0045,metal)
box('Tilt mechanism',(0,0,.044),(.022,.019,.007),metal,.0008)
for side in [-1,1]:
 rod('Arm support',(.024*side,.004,.05),(.027*side,.004,.068),.0023,metal)
 box('Armrest',(.027*side,-.006,.069),(.007,.029,.005),upholstery,.0013)
for index in range(int(settings['wheels'])):
 angle=index*2*math.pi/settings['wheels'];direction=Vector((math.sin(angle),math.cos(angle),0));end=direction*.033
 rod(f'Star branch {index+1}',(0,0,.02),(end.x,end.y,.011),.0024,metal)
 rod(f'Caster stem {index+1}',(end.x,end.y,.011),(end.x,end.y,.006),.0017,metal)
 tangent=Vector((direction.y,-direction.x,0))
 for offset in [-.0022,.0022]:
  center=Vector((end.x,end.y,.0048))+tangent*offset
  bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=.0047,depth=.003,location=center)
  wheel=bpy.context.object;wheel.rotation_euler=tangent.to_track_quat('Z','Y').to_euler();finish(wheel,f'Caster tire {index+1}',rubber,.0005)
 box(f'Caster fork {index+1}',(end.x,end.y,.008),(.004,.004,.004),metal,.0004)
# Scale the structural prior to the user-requested longest edge.
scale=float(settings['width_mm'])/122
for obj in parts:obj.scale*=scale;obj.location*=scale
bpy.ops.object.select_all(action='DESELECT')
for obj in parts:obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(folder/'model.glb'),export_format='GLB',use_selection=True)
(folder/'structure-parts.json').write_text(json.dumps({'parts':[obj.name for obj in parts],'through_holes':settings['holes'],'caster_groups':settings['wheels'],'all_dimensions_inferred':True},ensure_ascii=False,indent=2),encoding='utf-8')
exec(compile(Path(__file__).with_name('blender_ai_refine.py').read_text(encoding='utf-8'),str(Path(__file__).with_name('blender_ai_refine.py')),'exec'))
