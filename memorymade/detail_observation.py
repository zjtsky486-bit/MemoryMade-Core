"""Close-up observations are candidates, never certificates of 3D attachments."""
import base64
import io
import json
import urllib.request
from PIL import ImageOps


def needs_detail_review(description):
    person=str(description.get('detection_prompt','')).lower().strip()=='person'
    return person and any(isinstance(description.get(key),list) and description[key]
                          for key in ('thin_structures','held_objects'))


def detail_schema():
    text={'type':'string'}
    def obj(properties):
        return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    def coordinates(length):
        return {'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},'minItems':length,'maxItems':length}
    thin=obj({'part':text,'position':text,'box_1000':coordinates(4),'evidence_note':text})
    attachment=obj({'part':text,'endpoint':{'type':'string','enum':['a','b']},'connects_to':text,
                    'evidence':{'type':'string','enum':['visible','occluded','unknown']},
                    'point_1000':{'anyOf':[coordinates(2),{'type':'null'}]},'evidence_note':text})
    return obj({'object_name':text,'thin_structures':{'type':'array','items':thin,'maxItems':12},
                'attachments':{'type':'array','items':attachment,'maxItems':24},
                'uncertain_or_hidden':{'type':'array','items':text}})


def closeup_content(image,description=None):
    # A single targeted crop avoids asking a small VLM to resolve repeated
    # instances across a collage. Always retain the full-image context.
    positions=' '.join(str(item.get('position','')) for item in (description or {}).get('thin_structures',[]) if isinstance(item,dict)).lower()
    left=any(term in positions for term in ('左','left'));right=any(term in positions for term in ('右','right'))
    top=any(term in positions for term in ('上','top','upper'));bottom=any(term in positions for term in ('下','bottom','lower'))
    x1,x2=(0,.7) if left and not right else (.3,1) if right and not left else (0,1)
    y1,y2=(0,.7) if top and not bottom else (.3,1) if bottom and not top else (0,1)
    crop=image.crop((round(x1*image.width),round(y1*image.height),round(x2*image.width),round(y2*image.height)))
    crop=ImageOps.contain(crop,(1024,1024))
    prompt='''Inspect the visible thin accessories of this person. Image 1 is the full original; image 2 is a close-up of the SAME original (同一原图). They are not two objects.
Individually identify physical paper streamers, ropes, wires and thin rods. Name each distinct part specifically, with a unique suffix for separate instances. Do not label a whole quadrant or the entire held object as a thin structure. Do not include hair, wide sleeves, skirt edges or flat printed patterns.
Trace both endpoints a and b of EACH part. A visible free end connects_to 自由末端. An endpoint hidden behind an object is occluded; an untraceable endpoint is unknown. Do not infer that an accessory connects to a shoulder/body just because it is nearby. Never invent a hidden continuation.
box_1000 tightly encloses the individual part in the FULL ORIGINAL (原图中的) coordinates [x1,y1,x2,y2], normalized 0..1000 from upper-left to lower-right. Do not use crop coordinates or copy the entire crop as a part box.
point_1000 is [x,y] in the full original, or null if its location cannot be determined. A visible endpoint requires a point. Evidence describes the observed contact or free tip, not character lore. Be brief and use Chinese descriptions. Return only the specified JSON. No required number of accessories; do not duplicate a part seen in both images.'''
    content=[{'type':'text','text':prompt}]
    for title,frame in [('原图',image),('同一原图的局部放大',crop)]:
        frame=frame.copy();frame.thumbnail((1536,1536))
        stream=io.BytesIO();frame.convert('RGB').save(stream,format='JPEG',quality=98)
        content.extend([{'type':'text','text':title},{'type':'image_url','image_url':{
            'url':'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()}}])
    return content


def merge_detail_observations(description,review):
    description=dict(description)
    accepted=[];discarded=[]
    for candidate in review.get('thin_structures',[]):
        if not isinstance(candidate,dict):continue
        box=candidate.get('box_1000')
        if (not isinstance(box,list) or len(box)!=4 or
                any(type(value) is not int or not 0<=value<=1000 for value in box) or
                box[2]<=box[0] or box[3]<=box[1] or not str(candidate.get('part','')).strip() or
                candidate.get('part') in ('手持物','物体','主体','附属结构','细长部件')):
            discarded.append(candidate);continue
        if any(candidate['part']==old['part'] or box==old['box_1000'] for old in accepted):
            discarded.append(candidate);continue
        accepted.append({**candidate,'observation_is_verified':False})
    attachments=[]
    parts={item['part'] for item in accepted}
    for candidate in review.get('attachments',[]):
        if not isinstance(candidate,dict) or candidate.get('part') not in parts:continue
        candidate=dict(candidate);point=candidate.get('point_1000')
        if (point is not None and (not isinstance(point,list) or len(point)!=2 or
                any(type(value) is not int or not 0<=value<=1000 for value in point))):
            candidate['point_1000']=None;candidate['evidence']='unknown'
        if candidate.get('evidence') not in ('visible','occluded','unknown'):
            candidate['evidence']='unknown'
        if candidate['evidence']=='visible' and candidate.get('point_1000') is None:
            candidate['evidence']='unknown'
        if candidate.get('point_1000') is not None:
            box=next(item['box_1000'] for item in accepted if item['part']==candidate['part'])
            x,y=candidate['point_1000']
            if not (box[0]-30<=x<=box[2]+30 and box[1]-30<=y<=box[3]+30):
                candidate['evidence']='unknown'
                candidate['coordinate_consistency']='端点不在该部件框附近，需要核对'
        candidate['geometry_connection_verified']=False
        attachments.append(candidate)
    description['detail_review']={'status':'completed','accepted_candidates':len(accepted),
        'discarded_invalid_candidates':discarded,'raw_observations':review,
        'geometry_connections_verified':False,'part_count_verified':False,'endpoint_coordinates_verified':False,
        'notice':'局部识别仍是观察候选，放大不能补出原图不存在的信息；端点与三维连接需核对。'}
    if accepted:
        description['initial_thin_structures']=description.get('thin_structures',[])
        description['initial_attachments']=description.get('attachments',[])
        description['thin_structures']=accepted
        # Do not keep an earlier broad shoulder attachment as confirmed evidence.
        # Detailed endpoints supersede those candidates; original claims are retained.
        old_names={str(item.get('part','')) if isinstance(item,dict) else str(item)
                   for item in description['initial_thin_structures']}
        thin_words=('绳','飘带','纸带','流苏','电线','rope','ribbon','streamer','cord','wire')
        retained=[item for item in description['initial_attachments'] if isinstance(item,dict)
                  and item.get('part') not in old_names
                  and not any(word in str(item.get('part','')).lower() for word in thin_words)
                  and item.get('part') not in parts]
        description['attachments']=[*retained,*attachments]
    hidden=description.get('uncertain_or_hidden',[])
    if not isinstance(hidden,list):hidden=[str(hidden)]
    description['uncertain_or_hidden']=list(dict.fromkeys(str(value) for value in
        [*hidden,*review.get('uncertain_or_hidden',[]),description['detail_review']['notice']] if value not in ('无','none','')))
    description['attachment_evidence_is_verified']=False
    return description


def check_endpoints_against_foreground(description,foreground,original_size):
    """A point on alpha proves foreground presence, not the identity of a join."""
    import numpy as np
    from PIL import Image
    result=dict(description);detail=dict(result.get('detail_review',{}))
    image=Image.open(foreground).convert('RGBA')
    if tuple(image.size)!=tuple(original_size):
        detail['foreground_check']={'status':'skipped','reason':'原图与前景尺寸不一致，未比较坐标'}
        result['detail_review']=detail;return result
    alpha=np.asarray(image.getchannel('A'));radius=max(2,round(max(image.size)*.008))
    attachments=[];missing=0;present=0
    for original in result.get('attachments',[]):
        if not isinstance(original,dict):continue
        candidate=dict(original);point=candidate.get('point_1000')
        if point is not None and (not isinstance(point,list) or len(point)!=2 or
                                  any(type(value) not in (int,float) or not 0<=value<=1000 for value in point)):
            candidate['evidence']='unknown';candidate['point_1000']=None
            candidate['foreground_support']='invalid_coordinate';point=None
        if isinstance(point,list) and len(point)==2:
            x=round(point[0]*(image.width-1)/1000);y=round(point[1]*(image.height-1)/1000)
            supported=bool((alpha[max(0,y-radius):min(image.height,y+radius+1),max(0,x-radius):min(image.width,x+radius+1)]>8).any())
            candidate['foreground_support']='present' if supported else 'missing'
            if supported:present+=1
            else:
                missing+=1;candidate['evidence']='unknown'
                candidate['foreground_note']='预测端点附近没有分割到主体，可能坐标错误或细部漏分割，连接仍待核对。'
        candidate['geometry_connection_verified']=False
        attachments.append(candidate)
    result['attachments']=attachments
    detail['foreground_check']={'status':'completed','supported_endpoints':present,
                                 'unsupported_endpoints':missing,'geometry_verified':False}
    result['detail_review']=detail
    if missing:result['uncertain_or_hidden']=[*result.get('uncertain_or_hidden',[]),'部分预测端点与前景不符，不能按这些坐标自动搭桥或声称连接正确。']
    return result


def review_accessories(image,description,url):
    payload={'messages':[{'role':'user','content':closeup_content(image,description)}],
             'temperature':0,'max_tokens':3200,
             'response_format':{'type':'json_object','schema':detail_schema()}}
    request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:body=json.load(response)
    review=json.loads(body['choices'][0]['message']['content'])
    result=merge_detail_observations(description,review)
    result['detail_review']['backend']=description.get('backend','')
    return result
