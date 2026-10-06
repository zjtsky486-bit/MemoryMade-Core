from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "MEMORYMADE"
APP_VERSION = "1.0.0"

APP_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = Path(os.getenv("MEMORYMADE_ASSET_DIR", str(APP_ROOT / "assets")))
BRAND_DIR = ASSET_DIR / "brand"
SAMPLE_DIR = ASSET_DIR / "samples"
VENDOR_DIR = ASSET_DIR / "vendor"
DATA_DIR = Path(os.getenv("MEMORYMADE_DATA_DIR", APP_ROOT / "data")).resolve()
PROJECT_DIR = DATA_DIR / "projects"
GALLERY_DIR = DATA_DIR / "gallery"
SHARE_DIR = DATA_DIR / "share"
THUMB_DIR = DATA_DIR / "thumbnails"
EXPORT_DIR = DATA_DIR / "exports"
PUBLIC_BASE_URL = os.getenv("MEMORYMADE_PUBLIC_URL", "").rstrip("/")
_BLENDER_CANDIDATE = Path(os.getenv('MEMORYMADE_RESOURCE_ROOT', str(APP_ROOT.parent))) / 'tools' / 'blender-4.5.14' / 'blender-4.5.14-windows-x64' / 'blender.exe'
_BLENDER_D = Path('D:/1223456789/tools/blender-4.5.14/blender-4.5.14-windows-x64/blender.exe')
if _BLENDER_D.exists():_BLENDER_CANDIDATE=_BLENDER_D
BLENDER_EXE = os.getenv('MEMORYMADE_BLENDER', str(_BLENDER_CANDIDATE) if _BLENDER_CANDIDATE.exists() else '')
GENERATOR_WEBHOOK = os.getenv("MEMORYMADE_GENERATOR_WEBHOOK", "").strip()
GENERATOR_API_KEY = os.getenv("MEMORYMADE_GENERATOR_API_KEY", "").strip()
COLMAP_EXE = os.getenv("COLMAP_EXE", "").strip()

for _path in (DATA_DIR, PROJECT_DIR, GALLERY_DIR, SHARE_DIR, THUMB_DIR, EXPORT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

MATERIALS = {
    "光敏树脂": {
        "key": "resin",
        "color": "#E9E4D8",
        "roughness": 0.24,
        "metalness": 0.03,
        "description": "细腻、适合人物与细节丰富的记忆浮雕。",
    },
    "尼龙": {
        "key": "nylon",
        "color": "#B8BBC0",
        "roughness": 0.72,
        "metalness": 0.02,
        "description": "轻、耐用，适合复杂结构和日常摆件。",
    },
    "胡桃木": {
        "key": "wood",
        "color": "#9B663F",
        "roughness": 0.82,
        "metalness": 0.0,
        "description": "温润的木质质感，适合纪念底座与刻字。",
    },
    "陶土": {
        "key": "clay",
        "color": "#B86C4B",
        "roughness": 0.88,
        "metalness": 0.0,
        "description": "带手作痕迹的暖色表面，感性而克制。",
    },
    "金属感": {
        "key": "metal",
        "color": "#C9A45C",
        "roughness": 0.28,
        "metalness": 0.76,
        "description": "适合作为奖章、纪念徽章与金色高光。",
    },
    "自定义": {
        "key": "custom",
        "color": "#D8B36A",
        "roughness": 0.48,
        "metalness": 0.08,
        "description": "由你决定最终颜色与表面气质。",
    },
}

MODES = {
    'AI 精细三维（Hunyuan3D）': 'Grounding DINO 定位、BiRefNet 前景分割、Hunyuan3D-2 mini 扩散重建，再交给 Blender。此模式生成几何与所选单色材质，未重建照片纹理；背面属于推测。',
    'AI 精细三维（Hunyuan3D 2.1）': 'Hunyuan3D-2.1 Shape 单图扩散重建，几何细节优于 2.0 mini；仍使用所选单色材质，不生成照片 PBR 纹理。',
    'AI 精细三维 + PBR（Hunyuan3D 2.1）': 'Hunyuan3D-2.1 Shape + PBR Paint，同时生成几何和 albedo / metallic-roughness 材质；需要已编译官方 CUDA 光栅化扩展。',
    'AI 完整三维（TripoSR）': 'Qwen 描述物体、YOLO-World 定位、去背景后生成完整网格，再由本地 Blender 保存模型。单图背面是推测，最长边按尺寸等比例缩放。',
    '物体结构重建': '从单图估计物体轮廓和比例，为凳子、椅子、桌子生成座面、腿和支撑结构的真实三维网格。',
    "记忆浮雕": "单图或多图生成可打印的立体浮雕，稳定、清晰，最适合实体纪念物。",
    "记忆叠层": "把 2–6 张照片组合成错落展开的立体记忆场景。",
    "完整三维": "通过配置的外部生成模型，从单图生成可旋转的完整 3D 资产。",
    "多视角重建": "为 20 张以上环拍照片准备重建流程，连接 COLMAP 后可输出真实结构。",
}

