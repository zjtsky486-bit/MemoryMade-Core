"""Closed ornament cutters using Shapely/GEOS and Mapbox Earcut."""
import math,json
def motif_paths(pattern):
    if pattern.startswith('回纹'):
        return [[(x,-.5),(x+.45,-.5),(x+.45,.5),(x,.5),(x,-.2),(x+.28,-.2),(x+.28,.2),(x+.12,.2)] for x in [-.9,-.3,.3,.9]]
    if pattern.startswith('祥云'):
        paths=[[(cx+.25*(1-t/80)*math.cos(t*math.pi/14),.15+.7*(1-t/80)*math.sin(t*math.pi/14)) for t in range(65)] for cx in [-.65,0,.65]]
        return paths+[[(-1+2*t/80,-.65+.13*math.sin(t*math.pi/10)) for t in range(81)]]
    return [[(cx+.3*math.sin(t*math.tau/80),-.6+1.2*math.sin(t*math.pi/80)) for t in range(81)] for cx in [-.66,0,.66]]
def prepare_geometry(folder,pattern):
    from shapely import LineString,unary_union
    import trimesh
    polygons=unary_union([LineString(points).buffer(.025,quad_segs=6) for points in motif_paths(pattern)])
    bounds=polygons.bounds;cx=(bounds[0]+bounds[2])/2;cy=(bounds[1]+bounds[3])/2;rx=(bounds[2]-bounds[0])/2;ry=(bounds[3]-bounds[1])/2
    pieces=list(polygons.geoms) if hasattr(polygons,'geoms') else [polygons]
    data=[]
    for polygon in pieces:
        mesh=trimesh.creation.extrude_polygon(polygon,height=1,engine='earcut')
        if not mesh.is_watertight:raise ValueError('纹样切割体没有封闭。')
        vertices=mesh.vertices.copy();vertices[:,0]=(vertices[:,0]-cx)/rx;vertices[:,1]=(vertices[:,1]-cy)/ry
        data.append({'vertices':vertices.tolist(),'faces':mesh.faces.tolist()})
    (folder/'ornament-cutter.json').write_text(json.dumps(data),encoding='utf-8')
