from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps
from scipy import ndimage
from .recognition import classify_image
from .vision_stack import depth_heightmap, segment_with_box


def _load_rgb(path: str | Path, max_side: int = 640) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return image


def _foreground_mask(image: Image.Image) -> np.ndarray:
    small = image.resize((192, 192), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)
    border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)], axis=0)
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))
    threshold = max(22.0, float(np.percentile(distance, 58)) * 0.66)
    mask = distance > threshold
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        mask = labels == keep
    if mask.sum() < mask.size * 0.035:
        return np.ones_like(mask, dtype=bool)
    return mask


def analyze_structure(image: Image.Image) -> dict:
    mask = _foreground_mask(image)
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return {'type': '通用物体', 'score': 0.0, 'width_ratio': 1.0, 'upper_lower_ratio': 1.0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return {'type': '通用物体', 'score': 0.0, 'width_ratio': 1.0, 'upper_lower_ratio': 1.0}
    widths = crop.sum(axis=1).astype(np.float32)
    upper = widths[int(crop.shape[0] * .22):max(int(crop.shape[0] * .52), int(crop.shape[0] * .22) + 1)]
    lower = widths[int(crop.shape[0] * .62):max(int(crop.shape[0] * .95), int(crop.shape[0] * .62) + 1)]
    upper_ratio = float(np.median(upper) / max(np.median(lower), 1.0)) if len(upper) and len(lower) else 1.0
    aspect = float(crop.shape[0] / max(crop.shape[1], 1))
    seat_score = float(np.clip((upper_ratio - .98) / .65, 0, 1) * np.clip((aspect - .45) / 1.1, 0, 1))
    width_ratio = float((x1 - x0 + 1) / max(image.width, 1))
    if seat_score >= .32 and upper_ratio >= 1.08:
        object_type = '凳子'
    elif aspect > .85 and upper_ratio > 1.02:
        object_type = '椅子'
    else:
        object_type = '通用物体'
    return {'type': object_type, 'score': round(seat_score, 3), 'width_ratio': round(width_ratio, 3), 'upper_lower_ratio': round(upper_ratio, 3)}


def _frustum(radius_bottom: float, radius_top: float, height: float, sections: int = 24) -> trimesh.Trimesh:
    angles = np.linspace(0.0, math.tau, int(sections), endpoint=False)
    bottom = np.column_stack([np.cos(angles) * radius_bottom, np.sin(angles) * radius_bottom, np.zeros_like(angles)])
    top = np.column_stack([np.cos(angles) * radius_top, np.sin(angles) * radius_top, np.full_like(angles, height)])
    vertices = np.vstack([bottom, top])
    faces = []
    for index in range(int(sections)):
        nxt = (index + 1) % int(sections)
        faces.append([index, nxt, int(sections) + nxt])
        faces.append([index, int(sections) + nxt, int(sections) + index])
    center_bottom = len(vertices)
    center_top = center_bottom + 1
    vertices = np.vstack([vertices, [[0, 0, 0], [0, 0, height]]])
    for index in range(int(sections)):
        nxt = (index + 1) % int(sections)
        faces.append([center_bottom, nxt, index])
        faces.append([center_top, int(sections) + index, int(sections) + nxt])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)

def _tint_mesh(mesh: trimesh.Trimesh, color: str, factor: float = 1.0) -> trimesh.Trimesh:
    value = color.strip().lstrip('#')
    if len(value) != 6:
        value = '9B663F'
    base = np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
    base = np.clip(base * factor, 0, 255).astype(np.uint8)
    alpha = np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.hstack([np.tile(base, (len(mesh.vertices), 1)), alpha]))
    return mesh


def _average_color(image: Image.Image, fallback: str) -> str:
    small = np.asarray(image.resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float32)
    mask = _foreground_mask(image)
    mask_small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((48, 48), Image.Resampling.NEAREST), dtype=np.float32) > 80
    if mask_small.any():
        avg = small[mask_small].mean(axis=0)
    else:
        avg = small.reshape(-1, 3).mean(axis=0)
    if avg.max() < 55:
        return fallback
    return '#%02X%02X%02X' % tuple(int(value) for value in np.clip(avg, 0, 255))


def build_furniture(image_path: str | Path, kind: str, width_mm: float, height_mm: float, color: str) -> trimesh.Trimesh:
    image = _load_rgb(image_path)
    recognized = classify_image(image_path)
    detections = recognized.get('yolo', {}).get('detections', [])
    structural = next((item for item in detections if item.get('type') in {'凳子', '椅子', '桌子'}), None)
    if structural:
        x1, y1, x2, y2 = structural['box']
        mask = segment_with_box(image_path, structural['box'])
        if mask is not None:
            background = Image.new('RGB', image.size, (248, 245, 238))
            image = Image.composite(image, background, Image.fromarray(mask.astype(np.uint8) * 255))
        pad_x = (x2 - x1) * 0.08
        pad_y = (y2 - y1) * 0.08
        crop_box = (int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)), int(min(image.width, x2 + pad_x)), int(min(image.height, y2 + pad_y)))
        if crop_box[2] - crop_box[0] > 20 and crop_box[3] - crop_box[1] > 20:
            image = image.crop(crop_box)
    metrics = analyze_structure(image)
    if kind == '自动识别':
        if metrics.get('type') in {'凳子', '椅子', '桌子'}:
            actual_kind = metrics['type']
        else:
            actual_kind = recognized.get('semantic_type', '凳子')
            if actual_kind not in {'凳子', '椅子', '桌子'}:
                actual_kind = '凳子'
    else:
        actual_kind = kind
    if actual_kind not in {'凳子', '椅子', '桌子'}:
        actual_kind = '凳子'
    width_mm = float(np.clip(width_mm, 60, 220))
    height_mm = float(np.clip(height_mm, 60, 220))
    aspect = image.width / max(image.height, 1)
    depth = depth_heightmap(image)
    depth_span = float(np.percentile(depth, 90) - np.percentile(depth, 10)) if depth is not None else 0.5
    seat_width = width_mm * (0.94 if actual_kind != '桌子' else 0.88)
    seat_depth = seat_width * (0.74 if aspect > 1.15 else 0.92) * float(np.clip(0.88 + depth_span * 0.35, 0.82, 1.12))
    seat_thickness = max(3.0, height_mm * (0.075 if actual_kind != '桌子' else 0.045))
    overlap = max(0.8, seat_thickness * 0.4)
    leg_height = max(12.0, height_mm - seat_thickness + overlap)
    leg_radius = max(1.6, min(seat_width, seat_depth) * (0.045 if actual_kind != '桌子' else 0.035))
    seat_z = leg_height - overlap + seat_thickness / 2.0
    parts: list[trimesh.Trimesh] = []
    if aspect > 1.18 and actual_kind != '凳子':
        seat = trimesh.creation.box(extents=[seat_width, seat_depth, seat_thickness])
    else:
        seat = trimesh.creation.cylinder(radius=seat_width / 2.0, height=seat_thickness, sections=64)
    seat.apply_translation([0, 0, seat_z])
    parts.append(_tint_mesh(seat, color, 1.0))
    if aspect > 1.18:
        leg_x = seat_width / 2.0 - leg_radius * 2.0
        leg_y = seat_depth / 2.0 - leg_radius * 2.0
        leg_positions = [(-leg_x, -leg_y), (leg_x, -leg_y), (-leg_x, leg_y), (leg_x, leg_y)]
    else:
        radius = seat_width * 0.31
        leg_positions = [(math.cos(angle) * radius, math.sin(angle) * radius) for angle in (0.785, 2.356, 3.927, 5.498)]
    for x, y in leg_positions:
        leg = _frustum(leg_radius * 1.08, leg_radius * 0.78, leg_height, sections=22)
        leg.apply_translation([x, y, 0.0])
        parts.append(_tint_mesh(leg, color, 0.88))
        foot = trimesh.creation.cylinder(radius=leg_radius * 1.24, height=1.4, sections=24)
        foot.apply_translation([x, y, 0.7])
        parts.append(_tint_mesh(foot, color, 0.72))
    cross_radius = max(1.05, leg_radius * 0.68)
    side_y = seat_depth * 0.34
    side_x = seat_width * 0.34
    for cross_z in (leg_height * 0.38, leg_height * 0.68):
        for x in (-side_x, side_x):
            bar = trimesh.creation.cylinder(radius=cross_radius, height=side_y * 2.05, sections=16)
            bar.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0]))
            bar.apply_translation([x, 0, cross_z])
            parts.append(_tint_mesh(bar, color, 0.76))
        for y in (-side_y, side_y):
            bar = trimesh.creation.cylinder(radius=cross_radius, height=side_x * 2.05, sections=16)
            bar.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0]))
            bar.apply_translation([0, y, cross_z])
            parts.append(_tint_mesh(bar, color, 0.76))
    if actual_kind == '椅子':
        back_height = leg_height * 0.85
        back = trimesh.creation.box(extents=[seat_width * 0.94, max(3.0, seat_depth * 0.09), back_height])
        back.apply_translation([0, -seat_depth * 0.43, seat_z + back_height / 2.0])
        parts.append(_tint_mesh(back, color, 0.9))
    try:
        mesh = trimesh.boolean.union(parts, engine='manifold')
    except Exception:
        mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda component: len(component.vertices))
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    mesh.merge_vertices()
    mesh.metadata['name'] = f'MEMORYMADE {actual_kind}'
    mesh.metadata['object_type'] = actual_kind
    mesh.metadata['image_color'] = _average_color(image, color)
    return mesh









