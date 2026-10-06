from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'mobilenetv2'
MODEL_PATH = MODEL_DIR / 'model_int8.onnx'
CONFIG_PATH = MODEL_DIR / 'config.json'
CLIP_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'clip-vit-base-patch32'
CLIP_MODEL_PATH = CLIP_DIR / 'model_int8.onnx'
CLIP_TOKENIZER_PATH = CLIP_DIR / 'tokenizer.json'
YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'yolov8n' / 'yolov8n.onnx'
COCO_LABELS = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
CLIP_PROMPTS = [
    ('a photo of a stool without a backrest', '凳子', '凳子'),
    ('a photo of a wooden stool', '凳子', '凳子'),
    ('a photo of a small seat with four legs', '凳子', '凳子'),
    ('a photo of a round stool', '凳子', '凳子'),
    ('a photo of a chair with a backrest', '椅子', '椅子'),
    ('a photo of a dining chair', '椅子', '椅子'),
    ('a photo of an armchair', '椅子', '椅子'),
    ('a photo of a bench', '长凳', '凳子'),
    ('a photo of a park bench', '长凳', '凳子'),
    ('a photo of a table with legs', '桌子', '桌子'),
    ('a photo of a dining table', '桌子', '桌子'),
    ('a photo of a desk', '桌子', '桌子'),
    ('a photo of a vase', '花瓶', '花瓶'),
    ('a photo of a bottle', '瓶子', '瓶子'),
    ('a photo of a cup', '杯子', '杯子'),
    ('a photo of a lamp', '台灯', '台灯'),
    ('a photo of a toy', '玩具', '通用物体'),
    ('a photo of an animal', '动物', '通用物体'),
    ('a photo of a person', '人物', '通用物体'),
    ('a photo of a building', '建筑', '通用物体'),
    ('a photo of a car', '汽车', '通用物体'),
    ('a photo of a landscape', '风景', '通用物体'),
]

CHINESE_LABELS = {
    'folding chair': '折叠椅',
    'rocking chair': '摇椅',
    'barber chair': '椅类',
    'park bench': '长凳',
    'studio couch': '沙发',
    'dining table': '餐桌',
    'desk': '书桌',
    'table lamp': '台灯',
    'vase': '花瓶',
    'flowerpot': '花盆',
    'coffee mug': '咖啡杯',
    'cup': '杯子',
    'water bottle': '水瓶',
    'wine bottle': '酒瓶',
    'backpack': '背包',
    'handbag': '手提包',
    'teddy': '毛绒玩具',
    'screwdriver': '工具',
    'hammer': '工具',
    'laptop': '笔记本电脑',
    'cellular telephone': '手机',
    'camera': '相机',
    'clock': '时钟',
    'lamp': '灯具',
}

STRUCTURE_MAP = {
    '折叠椅': '椅子',
    '摇椅': '椅子',
    '椅类': '椅子',
    '长凳': '凳子',
    '餐桌': '桌子',
    '书桌': '桌子',
}

OBJECT_KEYWORDS = [
    ('folding chair', '椅子'),
    ('rocking chair', '椅子'),
    ('barber chair', '椅子'),
    ('park bench', '凳子'),
    ('dining table', '桌子'),
    ('desk', '桌子'),
    ('table lamp', '台灯'),
    ('vase', '花瓶'),
    ('flowerpot', '花盆'),
    ('coffee mug', '杯子'),
    ('cup', '杯子'),
    ('water bottle', '瓶子'),
    ('wine bottle', '瓶子'),
]


def _zh_label(label: str) -> str:
    lower = label.lower()
    for key, value in CHINESE_LABELS.items():
        if key in lower:
            return value
    return label.split(',')[0].strip()


def _semantic_type(label: str) -> str:
    lower = label.lower()
    for key, value in OBJECT_KEYWORDS:
        if key in lower:
            return value
    if 'chair' in lower:
        return '椅子'
    if 'bench' in lower or 'couch' in lower:
        return '凳子'
    if 'table' in lower or 'desk' in lower:
        return '桌子'
    if 'bottle' in lower:
        return '瓶子'
    if 'vase' in lower or 'pot' in lower:
        return '花瓶'
    if 'cup' in lower or 'mug' in lower:
        return '杯子'
    if 'lamp' in lower:
        return '台灯'
    return '通用物体'


@lru_cache(maxsize=1)
def _load_model():
    import onnxruntime as ort
    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    labels = {int(index): label for index, label in config.get('id2label', {}).items()}
    return session, labels


def _preprocess(path: str | Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    width, height = image.size
    scale = 256.0 / max(min(width, height), 1)
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    left = max(0, (image.width - 224) // 2)
    top = max(0, (image.height - 224) // 2)
    image = image.crop((left, top, left + 224, top + 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def _classify_mobilenet(path: str | Path, top_k: int = 5) -> dict:
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        return {'ok': False, 'top1': '未知', 'semantic_type': '通用物体', 'confidence': 0.0, 'top5': []}
    session, labels = _load_model()
    logits = session.run(['logits'], {'pixel_values': _preprocess(path)})[0][0]
    # The ONNX classifier includes a shifted background class at index 0.
    logits = logits[1:1001] if len(logits) > 1000 else logits
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    order = np.argsort(probabilities)[::-1][:max(1, int(top_k))]
    results = []
    for index in order:
        label = labels.get(int(index) + 1, labels.get(int(index), 'Unknown'))
        results.append({'label': label, 'label_zh': _zh_label(label), 'score': round(float(probabilities[index]), 4), 'type': _semantic_type(label)})
    best = results[0]
    return {'ok': True, 'top1': best['label'], 'top1_zh': best['label_zh'], 'semantic_type': best['type'], 'confidence': best['score'], 'top5': results}



@lru_cache(maxsize=1)
def _load_clip_model():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    session = ort.InferenceSession(str(CLIP_MODEL_PATH), providers=['CPUExecutionProvider'])
    tokenizer = Tokenizer.from_file(str(CLIP_TOKENIZER_PATH))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=49407, pad_token='<|endoftext|>')
    return session, tokenizer


def _clip_preprocess(path: str | Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    width, height = image.size
    scale = 224.0 / max(min(width, height), 1)
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
    left = max(0, (image.width - 224) // 2)
    top = max(0, (image.height - 224) // 2)
    image = image.crop((left, top, left + 224, top + 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def _classify_clip(path: str | Path) -> dict:
    if not CLIP_MODEL_PATH.exists() or not CLIP_TOKENIZER_PATH.exists():
        return {'ok': False, 'top1_zh': '未知', 'semantic_type': '通用物体', 'confidence': 0.0, 'top5': []}
    session, tokenizer = _load_clip_model()
    prompts = [item[0] for item in CLIP_PROMPTS]
    encodings = tokenizer.encode_batch(prompts)
    input_ids = np.array([item.ids for item in encodings], dtype=np.int64)
    attention_mask = np.array([item.attention_mask for item in encodings], dtype=np.int64)
    logits = session.run(['logits_per_image'], {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': _clip_preprocess(path)})[0][0]
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    groups = {}
    for index, (prompt, label_zh, semantic) in enumerate(CLIP_PROMPTS):
        item = groups.setdefault(label_zh, {'score': 0.0, 'type': semantic, 'prompt': prompt})
        item['score'] += float(probabilities[index])
    ordered = sorted(groups.items(), key=lambda item: item[1]['score'], reverse=True)[:5]
    results = [{'label': data['prompt'], 'label_zh': label_zh, 'score': round(data['score'], 4), 'type': data['type']} for label_zh, data in ordered]
    best = results[0]
    return {'ok': True, 'top1': best['label'], 'top1_zh': best['label_zh'], 'semantic_type': best['type'], 'confidence': best['score'], 'top5': results, 'backend': 'clip'}

@lru_cache(maxsize=1)
def _load_yolo_model():
    import onnxruntime as ort
    return ort.InferenceSession(str(YOLO_MODEL_PATH), providers=['CPUExecutionProvider'])


def _yolo_preprocess(path: str | Path):
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    width, height = image.size
    scale = min(640.0 / max(width, 1), 640.0 / max(height, 1))
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (640, 640), (114, 114, 114))
    pad_x = (640 - resized.width) // 2
    pad_y = (640 - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32), scale, pad_x, pad_y, width, height


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = max(0.0, boxes[index, 2] - boxes[index, 0]) * max(0.0, boxes[index, 3] - boxes[index, 1])
        area_b = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_a + area_b - inter, 1e-6)
        order = rest[iou <= threshold]
    return keep


def _classify_yolo(path: str | Path) -> dict:
    if not YOLO_MODEL_PATH.exists():
        return {'ok': False, 'detections': []}
    session = _load_yolo_model()
    image_array, scale, pad_x, pad_y, width, height = _yolo_preprocess(path)
    prediction = session.run(['output0'], {'images': image_array})[0][0].transpose(1, 0)
    class_scores = prediction[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]
    selected = np.where(confidences >= 0.25)[0]
    if len(selected) == 0:
        return {'ok': True, 'detections': []}
    centers = prediction[selected, :4]
    boxes = np.empty((len(selected), 4), dtype=np.float32)
    boxes[:, 0] = centers[:, 0] - centers[:, 2] / 2
    boxes[:, 1] = centers[:, 1] - centers[:, 3] / 2
    boxes[:, 2] = centers[:, 0] + centers[:, 2] / 2
    boxes[:, 3] = centers[:, 1] + centers[:, 3] / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    scores = confidences[selected]
    detections = []
    for index in _nms(boxes, scores)[:10]:
        label = COCO_LABELS[int(class_ids[selected[index]])]
        detections.append({'label': label, 'label_zh': _zh_label(label), 'score': round(float(scores[index]), 4), 'type': _semantic_type(label), 'box': [round(float(value), 1) for value in boxes[index]]})
    return {'ok': True, 'detections': detections}

def classify_image(path: str | Path, top_k: int = 5) -> dict:
    mobile = _classify_mobilenet(path, top_k=top_k)
    try:
        clip = _classify_clip(path)
    except Exception:
        clip = {'ok': False, 'top1_zh': '未知', 'semantic_type': '通用物体', 'confidence': 0.0, 'top5': []}
    try:
        yolo = _classify_yolo(path)
    except Exception:
        yolo = {'ok': False, 'detections': []}
    structure_types = {'凳子', '椅子', '桌子'}
    yolo_structural = next((item for item in yolo.get('detections', []) if item.get('type') in structure_types), None)
    if yolo_structural:
        best = {'ok': True, 'top1': yolo_structural['label'], 'top1_zh': yolo_structural['label_zh'], 'semantic_type': yolo_structural['type'], 'confidence': yolo_structural['score'], 'top5': yolo.get('detections', [])[:5], 'backend': 'yolo'}
    elif clip.get('ok') and (clip.get('semantic_type') in structure_types or clip.get('confidence', 0) >= 0.16):
        best = clip
    elif mobile.get('ok') and mobile.get('semantic_type') in structure_types:
        best = mobile
    else:
        best = clip if clip.get('ok') else mobile
    best = dict(best)
    best['mobile'] = mobile
    best['clip'] = clip
    best['yolo'] = yolo
    best['top1_zh'] = best.get('top1_zh', '未知')
    best['semantic_type'] = best.get('semantic_type', '通用物体')
    from .ai_stack import analyze_object
    advanced = analyze_object(path)
    best['advanced'] = advanced
    if advanced.get('ok'):
        description = advanced.get('description', {})
        best['top1_zh'] = description.get('object_name', best['top1_zh'])
        best['backend'] = 'Qwen2.5-VL-3B + YOLO-World'
        # Language descriptions do not carry calibrated confidence scores.
        best['confidence'] = 0.0
        label = best['top1_zh']
        best['semantic_type'] = next((kind for token, kind in [('凳', '凳子'), ('椅', '椅子'), ('桌', '桌子')] if token in label), '通用物体')
        detections = advanced.get('detections', [])
        if detections:
            best['yolo'] = {'ok': True, 'detections': [dict(item, type=best['semantic_type'], label_zh=label) for item in detections]}
    return best
