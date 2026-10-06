"""Close collinear triangle T junctions without changing the surface geometry."""
import numpy as np
def close_t_junctions(mesh):
    for _ in range(16):
        counts=np.bincount(mesh.edges_unique_inverse)
        boundary=mesh.edges_unique[counts==1]
        if not len(boundary):return mesh
        candidates=np.unique(boundary)
        changed=False
        for a,b in boundary:
            start,end=mesh.vertices[[a,b]];delta=end-start;length=float(delta@delta)
            if length<1e-12:continue
            t=(mesh.vertices[candidates]-start)@delta/length
            distance=np.linalg.norm(mesh.vertices[candidates]-(start+t[:,None]*delta),axis=1)
            inside=(t>1e-7)&(t<1-1e-7)&(distance<1e-6)
            points=candidates[inside]
            if not len(points):continue
            rows=np.flatnonzero(np.any(mesh.faces==a,axis=1)&np.any(mesh.faces==b,axis=1))
            if len(rows)!=1:continue
            row=rows[0];face=mesh.faces[row];index=int(np.flatnonzero(face==a)[0])
            if face[(index+1)%3]!=b:a,b=b,a;index=int(np.flatnonzero(face==a)[0])
            c=face[(index+2)%3];delta=mesh.vertices[b]-mesh.vertices[a]
            points=sorted(points,key=lambda p:float((mesh.vertices[p]-mesh.vertices[a])@delta))
            chain=[a,*points,b];triangles=[[chain[i],chain[i+1],c] for i in range(len(chain)-1)]
            mesh.faces=np.concatenate([np.delete(mesh.faces,row,axis=0),np.asarray(triangles)])
            changed=True;break
        if not changed:break
    return mesh
