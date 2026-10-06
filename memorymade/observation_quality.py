"""Observation coverage is evidence for review, not geometry conditioning."""
FACES = ('front', 'left', 'back', 'right', 'top', 'bottom')
LABELS = dict(zip(FACES, ('正面', '左侧', '背面', '右侧', '顶面', '底面')))


def coverage_review(description, comparison, fused_views, subject_kind='object'):
    observed = set()
    records = comparison.get('view_observations', [])
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and isinstance(record.get('observed_faces'), list):
                observed.update(face for face in record['observed_faces'] if face in FACES)
    if isinstance(description.get('observed_faces'), list):
        observed.update(face for face in description['observed_faces'] if face in FACES)
    missing = [face for face in FACES if face not in observed]
    notice = '视角覆盖为视觉识别候选，仍需与原照片核对。'
    if subject_kind != 'human' and 'bottom' not in observed:
        notice += '底面未被照片确认，底部结构属于推测，请补拍底面核对。'
        if '鼠标' in str(description.get('object_name','')) or 'mouse' in str(description.get('detection_prompt','')).lower():notice+='鼠标的光学传感器、脚贴和底盖不能仅凭顶部照片确定。'
    elif 'bottom' in observed:
        notice += '已观察到底面参考图；当前混元多视角形状接口仅接收正面、左侧、背面、右侧，底面参考用于核对，没有冒充参与形状融合。'
    if subject_kind == 'human':
        notice += '手持物、绳子和手指的遮挡连接需补拍特写；网格连通不代表连接位置一定正确。'
    return {'observed_faces': [face for face in FACES if face in observed],
            'unconfirmed_faces': missing, 'shape_conditioning_views': list(fused_views),
            'observation_is_verified': False, 'notice': notice,
            'thin_structures': description.get('thin_structures', []),
            'attachments': description.get('attachments', []),
            'held_objects': description.get('held_objects', [])}
