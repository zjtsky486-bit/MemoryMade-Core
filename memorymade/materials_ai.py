"""Material appearance catalog and conservative recommendation rules."""
CATALOG=[
 ('碳纤维','carbon fiber with a black woven diagonal twill pattern','#292C30',.32,.15),
 ('凯夫拉纤维','yellow aramid kevlar woven composite fibers','#B5A15A',.65,0),
 ('玻璃纤维','white woven fiberglass reinforced composite','#C3C9C4',.48,0),
 ('环氧树脂','clear translucent cured epoxy resin','#CABEA4',.2,0),
 ('硅胶','soft matte silicone rubber','#BEC1C1',.7,0),
 ('橡胶','black rubber with a matte flexible surface','#292A2B',.85,0),
 ('泡沫','porous polyurethane foam sponge','#BFAD78',.88,0),
 ('软木','natural cork with speckled porous brown texture','#AD8A56',.9,0),
 ('竹材','natural bamboo with linear grain','#BCA06C',.65,0),
 ('皮革','leather upholstery with visible grain and seams','#4D3630',.55,0),
 ('仿皮革','synthetic faux leather or vinyl upholstery','#433D36',.5,0),
 ('麂皮','soft suede with a velvety matte nap','#8C7966',.9,0),
 ('绒布','velvet fabric with a soft directional pile','#745C54',.85,0),
 ('织物','woven textile fabric upholstery','#8D8A7E',.9,0),
 ('ABS塑料','hard molded ABS plastic surface','#777B80',.4,0),
 ('亚克力','transparent acrylic plastic sheet','#D1DFE1',.15,0),
 ('玻璃','clear reflective glass','#D5E5E3',.08,0),
 ('大理石','polished marble stone with veins','#D9D4C6',.25,0),
 ('花岗岩','speckled granite stone','#9B9990',.65,0),
 ('混凝土','rough gray concrete cement','#9B9B92',.92,0),
 ('砖石','red brick masonry','#A26549',.85,0),
 ('石膏','white plaster gypsum surface','#DDD8CF',.88,0),
 ('陶瓷','glazed ceramic pottery','#E2DCCC',.22,0),
 ('陶土','unglazed terracotta clay','#B86C4B',.88,0),
 ('木头','natural wood with visible wood grain','#96623D',.65,0),
 ('尼龙','matte nylon polymer','#B8BBC0',.72,0),
 ('光敏树脂','smooth opaque 3D printed photopolymer resin','#E9E4D8',.24,.03),
 ('金属','brushed or polished metal','#A9ADB1',.3,.88),
 ('黄铜','warm golden brass metal','#C49C49',.28,.9),
 ('铝合金','silver aluminum alloy metal','#BFC2C3',.32,.9),
 ('不锈钢','brushed stainless steel metal','#A9ACAC',.3,.9),
 ('钛合金','gray titanium metal alloy','#A7A5A0',.38,.85),
 ('铜','reddish copper metal','#B8734B',.3,.9),
 ('纸板','brown corrugated cardboard paper','#B99A70',.88,0),
 ('水','liquid water','#94BECD',.1,0),
 ('植被','green leaves foliage and plants','#59764A',.85,0),
 ('未知表面','an ambiguous material surface without identifiable texture','#B1ADA1',.55,0),
]
LOOKUP={row[0]:{'key':row[0],'color':row[2],'roughness':row[3],'metalness':row[4]} for row in CATALOG}
METAL_TERMS=['金属','黄铜','青铜','不锈钢','钛合金','铝合金','铜','铁','钢','银','黄金','metal','brass','bronze','steel','copper','aluminum','aluminium','titanium']

def is_metal(name):return any(term in str(name).lower() for term in METAL_TERMS)

def custom_parameters(name):
    key=next((label for label in sorted(LOOKUP,key=len,reverse=True) if label in name),None)
    if key is None and 'carbon' in name.lower():key='碳纤维'
    return LOOKUP.get(key)

def recommendation(report,qwen_description=None):
    candidate=report.get('material','未知表面')
    if candidate in ['光敏树脂','尼龙','木头','陶土','金属']:
        return candidate,''
    return '自定义',candidate if candidate!='未知表面' else '待确认材料'
