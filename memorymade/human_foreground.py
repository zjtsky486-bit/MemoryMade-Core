"""Preserve observed person/prop alpha pixels; never paint missing connections."""
import numpy as np
from scipy.ndimage import label, distance_transform_edt


def select_person_foreground(alpha, pose):
    labels, count = label(alpha > 8, structure=np.ones((3, 3), dtype=int))
    scores = np.bincount(labels.ravel(), weights=(pose > 80).ravel())
    scores[0] = 0
    if not scores.max():
        raise ValueError('人物骨架与前景无法对齐，请使用背景更干净的照片。')
    principal = int(scores.argmax())
    # Pose segmentation describes the body, not held toys, cords or accessories.
    # Retain the entire observed connected alpha component, not its pose intersection.
    keep = labels == principal
    distance = distance_transform_edt(~keep)
    near_limit = max(3, max(alpha.shape) * .035)
    retained, removed = [], []
    selected=[principal]
    counts = np.bincount(labels.ravel())
    overlaps = np.bincount(labels.ravel(), weights=(pose > 120).ravel())
    gaps = np.full(count + 1, np.inf)
    np.minimum.at(gaps, labels.ravel(), distance.ravel())
    for region in range(1, count + 1):
        if region == principal:
            continue
        entry = {'pixels': int(counts[region]), 'gap_pixels': round(float(gaps[region]), 2)}
        if overlaps[region] >= .5 * counts[region] or gaps[region] <= near_limit:
            selected.append(region)
            retained.append(entry)
        else:
            removed.append(entry)
    keep=np.isin(labels,selected)
    review = {'policy': 'preserve_connected_person_and_props_v2',
              'retained_nearby_regions': retained[:100], 'removed_remote_regions': removed[:100],
              'retained_nearby_count':len(retained),'removed_remote_count':len(removed),
              'principal_pixels': int(counts[principal]),
              'connection_verified': False, 'synthetic_connections_added': False}
    return np.where(keep, alpha, 0).astype('uint8'), review
