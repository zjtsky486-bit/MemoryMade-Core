from __future__ import annotations

import os
import subprocess
from pathlib import Path

import trimesh

from .config import BLENDER_EXE, EXPORT_DIR


def refine_with_blender(glb_path: str | Path, stl_path: str | Path, obj_path: str | Path) -> dict:
    blender = Path(BLENDER_EXE) if BLENDER_EXE else None
    script = Path(__file__).with_name('blender_refine.py')
    if not blender or not blender.exists() or not script.exists():
        return {'ok': False, 'reason': 'blender_not_configured'}

    glb_path = Path(glb_path)
    stl_path = Path(stl_path)
    obj_path = Path(obj_path)
    stem = glb_path.stem
    refined_glb = glb_path.with_name(f'{stem}-blender.glb')
    refined_stl = stl_path.with_name(f'{stem}-blender.stl')
    refined_obj = obj_path.with_name(f'{stem}-blender.obj')
    blend_path = glb_path.with_name(f'{stem}-blender.blend')
    log_path = Path(EXPORT_DIR) / 'blender_refine.log'
    command = [str(blender), '--background', '--python', str(script), '--', str(glb_path), str(refined_glb), str(refined_stl), str(refined_obj), str(blend_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        log_path.write_text((result.stdout or '') + '\n' + (result.stderr or ''), encoding='utf-8')
    except Exception as exc:
        log_path.write_text(str(exc), encoding='utf-8')
        return {'ok': False, 'reason': str(exc)}
    if refined_glb.exists():
        os.replace(refined_glb, glb_path)
    if refined_stl.exists():
        os.replace(refined_stl, stl_path)
    if refined_obj.exists():
        os.replace(refined_obj, obj_path)
    try:
        mesh = trimesh.load(str(glb_path), force='mesh')
        mesh.merge_vertices()
        mesh.remove_degenerate_faces()
        if not mesh.is_watertight:
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)
            mesh.merge_vertices()
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda component: len(component.vertices))
        mesh.export(str(glb_path), file_type='glb')
        mesh.export(str(stl_path), file_type='stl')
        mesh.export(str(obj_path), file_type='obj')
    except Exception as exc:
        log_path.write_text((log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else '') + '\nmesh repair failed: ' + str(exc), encoding='utf-8')
    return {'ok': glb_path.exists(), 'reason': 'ok'}


