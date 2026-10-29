"""Local pilot inference and immutable inspection provenance."""
from pathlib import Path
import hashlib,json,uuid,io
import numpy as np,joblib
from PIL import Image,ImageOps
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import Response
from pydantic import BaseModel,Field
from typing import Literal
from . import core
from .paired_cnn import align,pair_tensor

router=APIRouter(prefix='/api')
PROJECT=Path(__file__).resolve().parents[1]
ALLOWED=['kernel5','channels24','channels12','baseline','epochs60','lr0001']

def connection():
    c=core.db();c.executescript('''CREATE TABLE IF NOT EXISTS inspections(id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, request_hash TEXT NOT NULL, data TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS inspection_decisions(id TEXT PRIMARY KEY, inspection_id TEXT NOT NULL, data TEXT NOT NULL);''');return c

def references():
    root=PROJECT/'output/reference-images';path=root/'manifest.json'
    return json.loads(path.read_text())['items'] if path.exists() else []
def ref(rid):
    r=next((r for r in references() if r['id']==rid),None)
    if r is None:raise HTTPException(404,'기준 시안을 찾을 수 없습니다.')
    return r,PROJECT/'output/reference-images'/r['image']
def gray(path):
    with Image.open(path) as im:return np.asarray(ImageOps.exif_transpose(im).convert('L'),dtype=float)/255

def registry():
    report=PROJECT/'output/cnn-hyperparameters/report.json'
    models=json.loads(report.read_text())['models'] if report.exists() else []
    return [{'id':m['name'],'service':'plate','stage':'experimental','roi_size':64,'threshold':m['threshold'],'config':m['config'],'test':m['test']} for m in models if m['name'] in ALLOWED and (PROJECT/f"output/cnn-hyperparameters/{m['name']}.joblib").exists()]

class Region(BaseModel):
    x:int=Field(ge=0,strict=True)
    y:int=Field(ge=0,strict=True)
class Request(BaseModel):
    request_key:str=Field(min_length=8,max_length=100)
    service:Literal['plate','die']='plate'
    reference_id:str
    sample_id:str
    model_id:Literal['kernel5','channels24','channels12','baseline','epochs60','lr0001']='kernel5'
    reference_roi:Region
    sample_roi:Region
    lot:str=Field(default='',max_length=100)
    operator:str=Field(min_length=1,max_length=100)
    no_overlay_confirmed:bool
    experimental_ack:bool
class Decision(BaseModel):
    reviewer:str=Field(min_length=1,max_length=100)
    disposition:Literal['OK','NG','RECHECK']
    note:str=Field(default='',max_length=2000)

@router.get('/references')
def list_references():return {'items':references()}
@router.get('/references/{rid:path}/preview')
def reference_preview(rid:str):
    _,path=ref(rid)
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail((2400,2400));buf=io.BytesIO();im.save(buf,format='PNG')
    return Response(buf.getvalue(),media_type='image/png')
@router.get('/models')
def models():return {'items':registry(),'die_status':'not_trained'}

def get_record(iid):
    with connection() as c:r=c.execute('SELECT data FROM inspections WHERE id=?',(iid,)).fetchone()
    if not r:raise HTTPException(404,'검사 이력이 없습니다.')
    return json.loads(r['data'])

@router.post('/inspections')
def inspect(req:Request):
    if req.service!='plate':raise HTTPException(409,'목형 모델은 아직 학습되지 않았습니다.')
    if not req.operator.strip():raise HTTPException(422,'검사자를 입력하세요.')
    if not req.no_overlay_confirmed or not req.experimental_ack:raise HTTPException(422,'표시 없는 사진과 실험 모델 사용 조건을 확인하세요.')
    request=req.model_dump();digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    with connection() as c:
        row=c.execute('SELECT data,request_hash FROM inspections WHERE request_key=?',(req.request_key,)).fetchone()
        if row:
            if row['request_hash']!=digest:raise HTTPException(409,'같은 요청 키에 다른 입력을 사용할 수 없습니다.')
            return json.loads(row['data'])
    _,rp=ref(req.reference_id)
    try:s=core.sample(req.sample_id)
    except ValueError as e:raise HTTPException(404,str(e))
    if s.get('overlay_status')=='present':raise HTTPException(422,'결함 표시가 있는 사진은 사용할 수 없습니다.')
    mp=PROJECT/f'output/cnn-hyperparameters/{req.model_id}.joblib'
    if not mp.exists():raise HTTPException(409,'모델 파일이 준비되지 않았습니다.')
    ra,qa=gray(rp),gray(s['path']);tiles=[]
    for array,roi in [(ra,req.reference_roi),(qa,req.sample_roi)]:
        if roi.x+64>array.shape[1] or roi.y+64>array.shape[0]:raise HTTPException(422,'64×64 영역이 이미지 범위를 벗어납니다.')
        tiles.append(array[roi.y:roi.y+64,roi.x:roi.x+64])
    moved,quality=align(*tiles,max_shift=4);bundle=joblib.load(mp);score=None
    if quality['accepted']:score=float(bundle['model'].forward(np.array([pair_tensor(tiles[0],moved)]))[0][0])
    result={'id':str(uuid.uuid4()),'created_at':core.now(),'input':request,'model_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'sample_sha256':s['id'],'model_id':req.model_id,'threshold':bundle['threshold'],'score':score,'alignment':quality,
     'status':'inconclusive' if score is None else ('anomaly_candidate' if score>bundle['threshold'] else 'normal_candidate'),'scope':'selected_64px_roi_only','production_ready':False,'notice':'선택 영역의 실험 모델 점수입니다. 전체 문안 검사·출하 승인이 아닙니다.'}
    try:
        with connection() as c:c.execute('INSERT INTO inspections VALUES (?,?,?,?)',(result['id'],req.request_key,digest,json.dumps(result,ensure_ascii=False)))
    except __import__('sqlite3').IntegrityError:
        with connection() as c:row=c.execute('SELECT data,request_hash FROM inspections WHERE request_key=?',(req.request_key,)).fetchone()
        if row['request_hash']!=digest:raise HTTPException(409,'요청 충돌')
        return json.loads(row['data'])
    core.event('inspection',{'id':result['id'],'status':result['status']});return result
@router.get('/inspections')
def history(offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
    with connection() as c:
        total=c.execute('SELECT count(*) FROM inspections').fetchone()[0]
        rows=c.execute('SELECT data FROM inspections ORDER BY rowid DESC LIMIT ? OFFSET ?',(limit,offset)).fetchall()
    return {'total':total,'items':[json.loads(r['data']) for r in rows]}
@router.get('/inspections/{iid}')
def detail(iid:str):
    r=get_record(iid)
    with connection() as c:decisions=[json.loads(x['data']) for x in c.execute('SELECT data FROM inspection_decisions WHERE inspection_id=? ORDER BY rowid',(iid,))]
    return {**r,'decisions':decisions}
@router.post('/inspections/{iid}/decisions')
def decide(iid:str,body:Decision):
    get_record(iid)
    if not body.reviewer.strip():raise HTTPException(422,'검수자를 입력하세요.')
    data={**body.model_dump(),'id':str(uuid.uuid4()),'at':core.now()}
    with connection() as c:c.execute('INSERT INTO inspection_decisions VALUES (?,?,?)',(data['id'],iid,json.dumps(data,ensure_ascii=False)))
    core.event('inspection_decision',{'inspection_id':iid,**data});return data
