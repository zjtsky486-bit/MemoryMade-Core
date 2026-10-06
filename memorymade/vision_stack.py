from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEPTH_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'depth-anything-v2-small'
DEPTH_MODEL_PATH = DEPTH_DIR / 'model_quantized.onnx'
SAM_ENCODER_PATH = Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'mobile-sam' / 'MobileSam_SAMEncoder.onnx'
SAM_DECODER_PATH = Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'mobile-sam' / 'MobileSam_SAMDecoder.onnx'


@lru_cache(maxsize=1)
def _depth_session():
    import onnxruntime as ort
    return ort.InferenceSession(str(DEPTH_MODEL_PATH), providers=['CPUExecutionProvider'])


@lru_cache(maxsize=1)
def _sam_sessions():
    import onnxruntime as ort
    encoder = ort.InferenceSession(str(SAM_ENCODER_PATH), providers=['CPUExecutionProvider'])
    decoder = ort.InferenceSession(str(SAM_DECODER_PATH), providers=['CPUExecutionProvider'])
    return encoder, decoder


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    low = float(np.percentile(depth, 1))
    high = float(np.percentile(depth, 99))
    if high <= low:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - low) / (high - low), 0.0, 1.0)


def depth_heightmap(image: Image.Image) -> np.ndarray | None:
    if not DEPTH_MODEL_PATH.exists():
        return None
    source = ImageOps.exif_transpose(image).convert('RGB')
    original_size = source.size
    resized = source.resize((518, 518), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    tensor = np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)
    prediction = _depth_session().run(['predicted_depth'], {'pixel_values': tensor})[0][0]
    depth = _normalize_depth(prediction)
    near = 1.0 - depth
    near_image = Image.fromarray((near * 255).astype(np.uint8), mode='L').resize(original_size, Image.Resampling.BICUBIC)
    return np.asarray(near_image, dtype=np.float32) / 255.0


def segment_with_box(path: str | Path, box: list[float] | tuple[float, float, float, float]) -> np.ndarray | None:
    if not SAM_ENCODER_PATH.exists() or not SAM_DECODER_PATH.exists():
        return None
    source = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    width, height = source.size
    resized = source.resize((1024, 1024), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)
    encoder, decoder = _sam_sessions()
    embeddings = encoder.run(['image_embeddings'], {'image': tensor})[0]
    x1, y1, x2, y2 = box
    coords = np.array([[[x1 / max(width, 1) * 1024, y1 / max(height, 1) * 1024], [x2 / max(width, 1) * 1024, y2 / max(height, 1) * 1024]]], dtype=np.float32)
    labels = np.array([[2.0, 3.0]], dtype=np.float32)
    masks, scores = decoder.run(['masks', 'scores'], {'image_embeddings': embeddings, 'point_coords': coords, 'point_labels': labels})
    mask = masks[0, 0] > 0.0
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode='L').resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0
