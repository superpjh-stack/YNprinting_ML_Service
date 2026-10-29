"""Render reference artwork without asserting physical OK ground truth."""
import argparse,hashlib,json,html
from pathlib import Path
import pypdfium2 as pdf
from PIL import Image,ImageDraw

def export(sources,out,dpi=300):
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'previews').mkdir(exist_ok=True)
    rows=[]
    for i,source in enumerate(sources,1):
        source=Path(source);sha=hashlib.sha256(source.read_bytes()).hexdigest()
        doc=pdf.PdfDocument(source)
        for page_no in range(len(doc)):
            page=doc[page_no];name=f'ref_{i:03d}_p{page_no+1:02d}.png'
            bitmap=page.render(scale=dpi/72);im=bitmap.to_pil().convert('RGB');im.save(out/'images'/name,dpi=(dpi,dpi))
            preview=im.copy();preview.thumbnail((600,450));preview.save(out/'previews'/name)
            rows.append({'id':f'{sha}:{page_no+1}','source':str(source),'group':source.parent.name,'source_sha256':sha,'page':page_no+1,'image':'images/'+name,'preview':'previews/'+name,'dpi':dpi,'width':im.width,'height':im.height,'capture_kind':'artwork','label':None,'status':'reference_candidate','normal_capture':False,'cleanup_status':'unmodified_render','review_required':['시안 개정판과 제품 대응','재단선·분판·치수 포함 여부','인쇄용 정상 시안 승인 여부']})
            bitmap.close();page.close()
        doc.close()
    (out/'manifest.json').write_text(json.dumps({'version':'pdf-reference-v1','items':rows},ensure_ascii=False,indent=2))
    cards=''.join(f'<article><a href="{r["image"]}"><img src="{r["preview"]}" alt="시안 {i}"/></a><h2>{i:03d}</h2><p>{html.escape(r["group"])}</p><small>정상 참조 후보 · 검수 전 · 300 dpi</small></article>' for i,r in enumerate(rows,1))
    (out/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>와이엔프린팅 참조 시안</title><style>body{font-family:sans-serif;padding:30px;background:#edf1ed;color:#243f32}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}article{background:white;padding:16px}img{width:100%;height:260px;object-fit:contain}p{font-size:12px;overflow-wrap:anywhere}h2{font-size:18px}small{color:#746538}</style><h1>원본 PDF → 참조 이미지</h1><p>사진을 클릭하면 고해상도 PNG가 열립니다. 원본 시안의 인쇄 요소를 그대로 보존했습니다. 정상 실촬영 데이터나 승인된 OK 라벨이 아닙니다.</p><main>'+cards+'</main></html>')
    print(json.dumps({'exported':len(rows),'output':str(out)},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(json.loads(Path(a.sources).read_text()),Path(a.output))
