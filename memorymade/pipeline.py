from __future__ import annotations

import json
import math
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .config import EXPORT_DIR, MATERIALS
from .structures import build_furniture
from .vision_stack import depth_heightmap
from .blender_bridge import refine_with_blender

try:
    from scipy.ndimage import gaussian_filter, sobel
except Exception:
    gaussian_filter = None
    sobel = None


@dataclass
class ModelResult:
    project_id: str
    title: str
    mode: str
    material: str
    color: str
    glb_path: str
    stl_path: str
    obj_path: str
    preview_path: str
    source_images: list[str]
    stats: dict
    created_at: str

    def as_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=64)
def _grid_faces(ny: int, nx: int) -> np.ndarray:
    top_count = nx * ny
    y, x = np.mgrid[0:ny - 1, 0:nx - 1]
    a = (y * nx + x).ravel()
    b = a + 1
    c = a + nx
    d = c + 1
    top = np.column_stack([a, c, b, b, c, d]).reshape(-1, 3)
    bottom = np.column_stack([top_count + a, top_count + b, top_count + c, top_count + b, top_count + d, top_count + c]).reshape(-1, 3)
    x = np.arange(nx - 1)
    a = x
    b = a + 1
    edge_a = np.column_stack([a, b, top_count + b, a, top_count + b, top_count + a]).reshape(-1, 3)
    a = (ny - 1) * nx + x
    b = a + 1
    edge_b = np.column_stack([a, top_count + a, top_count + b, a, top_count + b, b]).reshape(-1, 3)
    y = np.arange(ny - 1)
    a = y * nx
    c = a + nx
    edge_c = np.column_stack([a, top_count + a, top_count + c, a, top_count + c, c]).reshape(-1, 3)
    a = y * nx + nx - 1
    c = a + nx
    edge_d = np.column_stack([a, c, top_count + c, a, top_count + c, top_count + a]).reshape(-1, 3)
    return np.vstack([top, bottom, edge_a, edge_b, edge_c, edge_d]).astype(np.int64)

def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_project_id(prefix: str = "mm") -> str:
    return f"{prefix}-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = (value or "#D8B36A").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        value = "D8B36A"
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _open_rgb(path: str | Path, max_side: int = 1400) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGBA")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _to_square(image: Image.Image, size: int, background=(22, 16, 11, 255)) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), background)
    canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def _edge_map(gray: np.ndarray) -> np.ndarray:
    if sobel is None:
        image = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
        return np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    magnitude = np.hypot(sobel(gray, axis=1), sobel(gray, axis=0))
    peak = float(np.percentile(magnitude, 98)) or 1.0
    return np.clip(magnitude / peak, 0.0, 1.0)


def estimate_heightmap(image: Image.Image, detail: float = 0.62, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    rgba = image.convert("RGBA")
    gray = np.asarray(rgba.convert("L"), dtype=np.float32) / 255.0
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    sigma = max(0.4, 3.6 * (1.0 - float(np.clip(detail, 0.0, 1.0))))
    if gaussian_filter is not None:
        base = gaussian_filter(gray, sigma=sigma)
    else:
        blurred = rgba.convert("L").filter(ImageFilter.GaussianBlur(radius=sigma))
        base = np.asarray(blurred, dtype=np.float32) / 255.0
    detail_layer = gray - base
    height = 0.64 * base + 0.28 * np.clip(detail_layer + 0.5, 0.0, 1.0) + 0.08 * _edge_map(gray)
    height = np.clip((height - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    if invert:
        height = 1.0 - height
    height = height * alpha
    if alpha.min() < 0.99:
        height = np.where(alpha > 0.04, height, 0.01)
    return height.astype(np.float32)


def _sample_heightmap(height: np.ndarray, resolution: int) -> np.ndarray:
    resolution = int(np.clip(resolution, 48, 220))
    image = Image.fromarray((np.clip(height, 0, 1) * 255).astype(np.uint8), mode="L")
    ratio = image.width / max(image.height, 1)
    if ratio >= 1:
        size = (resolution, max(32, int(round(resolution / ratio))))
    else:
        size = (max(32, int(round(resolution * ratio))), resolution)
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _mesh_from_heightmap(height: np.ndarray, color: str, width_mm: float, height_mm: float, relief_mm: float, base_mm: float, image: Image.Image | None = None) -> trimesh.Trimesh:
    height = _sample_heightmap(height, max(height.shape))
    ny, nx = height.shape
    xs = np.linspace(-width_mm / 2.0, width_mm / 2.0, nx, dtype=np.float32)
    ys = np.linspace(height_mm / 2.0, -height_mm / 2.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    top_count = nx * ny
    top = np.column_stack([xx.ravel(), yy.ravel(), (base_mm + height * relief_mm).ravel()])
    bottom = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(top_count, dtype=np.float32)])
    vertices = np.vstack([top, bottom]).astype(np.float64)
    faces = _grid_faces(ny, nx)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    base_rgb = np.asarray(hex_to_rgb(color), dtype=np.float32)
    sample = None
    if image is not None:
        sample = np.asarray(_to_square(image, max(nx, ny)).resize((nx, ny), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32) / 255.0
    top_colors = np.tile(base_rgb, (top_count, 1))
    if sample is not None:
        luminance = sample.mean(axis=2, keepdims=True)
        top_colors = np.clip(base_rgb.reshape(1, 1, 3) * (0.72 + 0.52 * luminance), 0, 255).reshape(-1, 3)
    bottom_colors = np.tile(base_rgb * 0.52, (top_count, 1))
    colors = np.vstack([top_colors, bottom_colors]).astype(np.uint8)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.hstack([colors, alpha]))
    mesh.metadata["name"] = "MEMORYMADE relief"
    return mesh


def _color_mesh(mesh: trimesh.Trimesh, color: str) -> trimesh.Trimesh:
    result = mesh.copy()
    base = np.asarray(hex_to_rgb(color), dtype=np.uint8)
    colors = getattr(result.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(result.vertices):
        vertex_colors = np.array(colors, copy=True)
        old = vertex_colors[:, :3].astype(np.float32)
        luminance = (0.2126 * old[:, 0] + 0.7152 * old[:, 1] + 0.0722 * old[:, 2]) / 255.0
        vertex_colors[:, :3] = np.clip(base.astype(np.float32) * (0.70 + 0.55 * luminance[:, None]), 0, 255).astype(np.uint8)
        result.visual.vertex_colors = vertex_colors
    else:
        alpha = np.full((len(result.vertices), 1), 255, dtype=np.uint8)
        result.visual = trimesh.visual.ColorVisuals(mesh=result, vertex_colors=np.hstack([np.tile(base, (len(result.vertices), 1)), alpha]))
    return result


def _single_relief(image_path: str | Path, color: str, width_mm: float, height_mm: float, relief_mm: float, base_mm: float, resolution: int, detail: float, contrast: float, invert: bool) -> tuple[trimesh.Trimesh, Image.Image]:
    source = _open_rgb(image_path)
    depth = depth_heightmap(source)
    if depth is not None:
        base_height = estimate_heightmap(source, detail=detail, contrast=contrast, invert=invert)
        height = np.clip(0.72 * depth + 0.28 * base_height, 0.0, 1.0)
    else:
        height = estimate_heightmap(source, detail=detail, contrast=contrast, invert=invert)
    sampled = _sample_heightmap(height, resolution)
    mesh = _mesh_from_heightmap(sampled, color, width_mm, height_mm, relief_mm, base_mm, source)
    return mesh, source


def _make_base(width_mm: float, depth_mm: float, height_mm: float, color: str) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[width_mm, depth_mm, height_mm])
    box.apply_translation([0, 0, height_mm / 2.0])
    return _color_mesh(box, color)


def combine_reliefs(meshes: Sequence[trimesh.Trimesh], canvas_width_mm: float = 150.0, layer_depth_mm: float = 4.0) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError("至少需要一张照片。")
    scene_depth = max(12.0, layer_depth_mm * max(1, len(meshes) - 1) + 12.0)
    parts = [_make_base(canvas_width_mm * 1.08, scene_depth, 4.0, "#6A4730")]
    count = len(meshes)
    for index, mesh in enumerate(meshes):
        part = mesh.copy()
        bounds = part.bounds
        part_width = float(bounds[1][0] - bounds[0][0])
        scale = min(0.80 if count == 1 else 0.62, (canvas_width_mm * 0.78) / max(part_width, 1.0))
        part.apply_scale(scale)
        part.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-72.0), [1, 0, 0]))
        if count == 1:
            x = 0.0
        else:
            x = (index - (count - 1) / 2.0) * (canvas_width_mm * 0.46 / max(count - 1, 1))
        y = (index - (count - 1) / 2.0) * layer_depth_mm
        z = 4.0 + index * 0.18
        part.apply_translation([x, y, z])
        parts.append(part)
    combined = trimesh.util.concatenate(parts)
    combined.merge_vertices()
    combined.remove_degenerate_faces()
    return combined


def mesh_stats(mesh: trimesh.Trimesh, width_mm: float, height_mm: float, relief_mm: float) -> dict:
    bounds = mesh.bounds
    size = np.maximum(bounds[1] - bounds[0], 0)
    try:
        volume = float(abs(mesh.volume))
    except Exception:
        volume = 0.0
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "size_mm": [round(float(v), 2) for v in size],
        "volume_cm3": round(volume / 1000.0, 2),
        "resolution": f"{width_mm:.0f} × {height_mm:.0f} × {relief_mm:.1f} mm",
    }


def create_preview_card(source_images: Sequence[str | Path], output_path: str | Path, title: str, subtitle: str, story: str, color: str, size: tuple[int, int] = (1280, 800)) -> Path:
    width, height = size
    start = np.array([12, 9, 6], dtype=np.float32)
    end = np.array([46, 32, 19], dtype=np.float32)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    gradient = (start[None, :] * (1.0 - t) + end[None, :] * t).astype(np.uint8)
    canvas = Image.fromarray(np.repeat(gradient[:, None, :], width, axis=1), mode='RGB').convert('RGBA')
    draw = ImageDraw.Draw(canvas, "RGBA")
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-260, -320, width + 100, 540), fill=(205, 151, 61, 70))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(90)))
    if source_images:
        source = _open_rgb(source_images[0], max_side=1800)
        cover = ImageOps.fit(source.convert("RGB"), (630, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.45)).convert("RGBA")
        cover.alpha_composite(Image.new("RGBA", cover.size, (4, 3, 2, 105)))
        canvas.alpha_composite(cover, (width - 630, 0))
        draw.rectangle([width - 632, 0, width - 1, height - 1], outline=(220, 172, 83, 180), width=1)
    accent = hex_to_rgb(color)
    draw.text((72, 62), "MEMORYMADE", font=_font(25, True), fill=(*accent, 255))
    draw.text((72, 101), "MAKE MEMORY TOUCHABLE", font=_font(12), fill=(168, 146, 109, 255))
    draw.text((72, 186), title[:18] or "未命名记忆", font=_font(50, True), fill=(242, 232, 207, 255))
    draw.text((74, 258), subtitle[:30], font=_font(22), fill=(213, 177, 105, 255))
    draw.multiline_text((74, 332), story[:180] or "把一段不想忘记的时间和影像，做成可以触摸、可以聆听的纪念物。", font=_font(22), fill=(224, 215, 195, 255), spacing=16)
    draw.line([(74, 492), (510, 492)], fill=(151, 112, 49, 180), width=1)
    draw.text((74, 522), "影像 · 建模 · 材质 · 声音", font=_font(18), fill=(208, 184, 141, 255))
    draw.text((74, height - 92), f"CREATED {datetime.now():%Y.%m.%d}", font=_font(13), fill=(153, 127, 82, 255))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, quality=94)
    return output_path


def export_model_bundle(result: ModelResult, audio_path: str | None = None, share_url: str = "", full: bool = True) -> Path:
    from .disk_guard import ensure_disk_space
    ensure_disk_space('导出制作包')
    archive = EXPORT_DIR / f'{result.project_id}-bundle.zip'
    if archive.resolve().parent != EXPORT_DIR.resolve():raise ValueError('无效导出路径')
    if not Path(result.glb_path).is_file():raise ValueError('原始模型缺失，无法导出制作包。')
    target = EXPORT_DIR / ('.bundle-' + uuid.uuid4().hex)
    temporary = target.with_suffix('.zip')
    target.mkdir(parents=True, exist_ok=False)
    try:
        if full and (not result.stl_path or not Path(result.stl_path).is_file() or not result.obj_path or not Path(result.obj_path).is_file()):
            try:
                mesh = load_mesh(result.glb_path)
                if not result.stl_path or not Path(result.stl_path).exists():
                    mesh.export(target/'model.stl', file_type='stl')
                if not result.obj_path or not Path(result.obj_path).exists():
                    mesh.export(target/'model.obj', file_type='obj')
            except Exception as exc:
                raise ValueError('打印格式导出失败，旧制作包与原模型已保留。') from exc
        for source in (result.glb_path, result.stl_path, result.obj_path, result.preview_path):
            if source and Path(source).exists():
                if Path(source).resolve()!=(target/Path(source).name).resolve():shutil.copy2(source, target / Path(source).name)
        metadata = result.as_dict()
        for key in ('glb_path','stl_path','obj_path','preview_path'):
            source = metadata.get(key)
            if source and (target/Path(source).name).is_file():metadata[key]=Path(source).name
        if (target/'model.stl').is_file() and not result.stl_path:metadata['stl_path']='model.stl'
        if (target/'model.obj').is_file() and not result.obj_path:metadata['obj_path']='model.obj'
        if full and 'Blender' in result.stats.get('backend',''):
            folder = Path(result.glb_path).parent
            for name in ['model.blend', 'recognition.json', 'generation.json', 'source.png', 'foreground.png', 'input.png', 'raw.glb', 'model.mtl', 'view-0.png', 'view-1.png', 'view-2.png']:
                source = folder / name
                if source.exists():
                    shutil.copy2(source, target / name)
            audio_sidecars={'transcript.json','transcript.txt','memory.ndef','memory-nfc-tlv.bin','nfc-address.txt','nfc-qr.png'}
            for source in folder.iterdir():
                if not audio_path and source.name in audio_sidecars:continue
                if source.is_file() and (source.name in ['transcript.json','transcript.txt','material.json','coin.json','photo.png','memory.ndef','memory-nfc-tlv.bin','nfc-address.txt','nfc-qr.png','view-map.json','identity.json','material-recognition.json','chair.json','structure-parts.json','scene-cameras.npz','carbon-weave.png','carbon-normal.png','ornament.json','ornament-report.json','ornament-cutter.json'] or source.name.startswith(('source-','foreground-','human-'))):
                    shutil.copy2(source,target/source.name)
        metadata["audio_file"] = Path(audio_path).name if audio_path else ""
        metadata["share_url"] = share_url
        (target / "project.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if audio_path and Path(audio_path).exists():
            shutil.copy2(audio_path, target / Path(audio_path).name)
        ensure_disk_space('压缩制作包')
        shutil.make_archive(str(target), 'zip', target)
        ensure_disk_space('完成制作包')
        temporary.replace(archive)
        return archive
    finally:
        temporary.unlink(missing_ok=True)
        if target.exists():shutil.rmtree(target)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    return trimesh.load(str(path), force="mesh")


def slugify(text: str, fallback: str = "memory") -> str:
    allowed = [char for char in text.strip().lower() if char.isalnum() or char in "-_"]
    value = "".join(allowed).strip("-_")
    return value[:42] or fallback


def generate_model(
    image_paths: Sequence[str | Path],
    mode: str,
    material: str,
    color: str,
    title: str,
    subtitle: str,
    story: str,
    width_mm: float = 120.0,
    height_mm: float = 90.0,
    relief_mm: float = 8.0,
    base_mm: float = 3.0,
    resolution: int = 128,
    detail: float = 0.62,
    contrast: float = 1.05,
    invert: bool = False,
    engraving: str = '',
    project_id: str | None = None,
    object_type: str = '自动识别',
) -> ModelResult:
    valid_images = []
    for image_path in image_paths:
        if not image_path or not Path(image_path).exists():
            continue
        try:
            with Image.open(image_path) as probe:
                probe.verify()
            valid_images.append(str(image_path))
        except Exception:
            continue
    if not valid_images:
        raise ValueError("请先上传至少一张照片。")
    project_id = project_id or new_project_id()
    work = Path(EXPORT_DIR) / project_id
    work.mkdir(parents=True, exist_ok=True)

    count = min(len(valid_images), 6)
    chosen = valid_images[:count]
    if mode == '物体结构重建':
        combined = build_furniture(chosen[0], object_type, width_mm, height_mm, color)
        combined.merge_vertices()
    else:
        def build_one(item):
            index, path = item
            mesh, _ = _single_relief(
                path,
                color,
                width_mm=width_mm,
                height_mm=height_mm,
                relief_mm=relief_mm,
                base_mm=base_mm,
                resolution=max(64, resolution - index * 4),
                detail=detail,
                contrast=contrast,
                invert=invert,
            )
            return mesh

        if count == 1:
            meshes = [build_one((0, chosen[0]))]
        else:
            workers = min(4, count)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mm-mesh') as pool:
                meshes = list(pool.map(build_one, enumerate(chosen)))
        if count == 1 and mode == '记忆浮雕':
            combined = meshes[0]
        else:
            combined = combine_reliefs(meshes, canvas_width_mm=max(width_mm, 110.0), layer_depth_mm=max(3.0, relief_mm * 0.55))
        if engraving.strip():
            plate = _engraving_plate(engraving, width_mm=min(width_mm * 0.62, 82.0), color=color)
            bounds = combined.bounds
            plate_height = float(plate.bounds[1][2] - plate.bounds[0][2])
            plate.apply_translation([0, float(bounds[0][1]) - 2.0, plate_height / 2.0])
            combined = trimesh.util.concatenate([combined, plate])
    combined.remove_degenerate_faces()
    stats = mesh_stats(combined, width_mm, height_mm, relief_mm)
    if mode == '物体结构重建':
        stats['object_type'] = combined.metadata.get('object_type', object_type)
    stem = f"{slugify(title, project_id)}-{project_id.split('-')[-1]}"
    glb_path = work / f"{stem}.glb"
    stl_path = work / f"{stem}.stl"
    obj_path = work / f"{stem}.obj"
    display = combined.copy()
    display.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0]))
    display.apply_translation(-display.bounds.mean(axis=0))
    display.export(glb_path, file_type="glb")
    combined.export(stl_path, file_type="stl")
    if mode == '物体结构重建':
        refine_with_blender(glb_path, stl_path, obj_path)
    preview_path = work / f"{stem}-preview.jpg"
    create_preview_card(chosen, preview_path, title=title, subtitle=subtitle, story=story, color=color)
    return ModelResult(
        project_id=project_id,
        title=title or "未命名记忆",
        mode=mode,
        material=material,
        color=color,
        glb_path=str(glb_path),
        stl_path=str(stl_path),
        obj_path=str(obj_path),
        preview_path=str(preview_path),
        source_images=chosen,
        stats=stats,
        created_at=now_stamp(),
    )



def _engraving_plate(text: str, width_mm: float, color: str) -> trimesh.Trimesh:
    clean = (text or '').strip()[:24]
    if not clean:
        return trimesh.Trimesh()
    width_px, height_px = 600, 150
    image = Image.new('L', (width_px, height_px), 0)
    draw = ImageDraw.Draw(image)
    font = _font(78, True)
    box = draw.textbbox((0, 0), clean, font=font)
    text_w = max(1, box[2] - box[0])
    text_h = max(1, box[3] - box[1])
    draw.text(((width_px - text_w) / 2 - box[0], (height_px - text_h) / 2 - box[1]), clean, font=font, fill=255)
    height = np.asarray(image, dtype=np.float32) / 255.0
    plate = _mesh_from_heightmap(height, color, width_mm=width_mm, height_mm=width_mm * height_px / width_px, relief_mm=1.8, base_mm=1.2, image=image.convert('RGB'))
    plate.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0]))
    return plate



















