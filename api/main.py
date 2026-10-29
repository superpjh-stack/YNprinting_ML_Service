from pathlib import Path
from typing import Literal
import json, tempfile, threading, uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import core, training, collection, annotations

app=FastAPI(title='와이엔프린팅 ML 작업실')
pool=ThreadPoolExecutor(max_workers=1); jobs={};lock=threading.Lock()
Service=Literal['plate','die','unassigned']
class Review(BaseModel):
    service:Service
    label:Literal['OK','NG']
    reviewer:str=Field(min_length=1,max_length=100)
    group:str=Field(min_length=1,max_length=300)
    domain:str=Field(min_length=1,max_length=100)
    boxes:list[annotations.Box]=Field(default_factory=list,max_length=200)
    annotation_complete:bool=False
    quality_accepted:bool=False
    overlay_status:Literal['unknown','present','absent']='unknown'
    capture_kind:Literal['unknown','plate_capture','die_capture','label_capture','artwork']='unknown'
class TrainingRequest(BaseModel): service:Literal['plate','die']

def submit(fn,*args):
    jid=str(uuid.uuid4());jobs[jid]={'id':jid,'status':'queued'}
    def work():
        jobs[jid]['status']='running'
        try:jobs[jid].update(status='succeeded',result=fn(*args))
        except Exception as e:jobs[jid].update(status='failed',error=str(e))
    pool.submit(work);return jobs[jid]

@app.get('/api/health')
def health():return {'status':'ok','model_ready':any((core.ROOT/'models'/r['id']/'model.joblib').is_file() for r in core.overview()['runs'])}
@app.get('/api/overview')
def overview():return core.overview()
@app.get('/api/samples')
def samples(service:Service|None=None,q:str='',offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
    rows=[s for s in core.all_samples() if (service is None or s['service']==service) and q.casefold() in (s['name']+' '+s['group']+' '+s['weak_label']).casefold()]
    return {'total':len(rows),'items':rows[offset:offset+limit]}
@app.post('/api/uploads')
async def upload(file:UploadFile=File(...),service:Service='unassigned'):
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in ['.png','.jpg','.jpeg']:raise HTTPException(422,'PNG/JPG 이미지를 선택하세요.')
    data=await file.read(30*1024*1024+1)
    if len(data)>30*1024*1024:raise HTTPException(413,'30MB 제한')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/('upload'+suffix);p.write_bytes(data)
        try:s,created=core.ingest(p,service)
        except Exception as e:raise HTTPException(422,str(e))
    if created:s['name']=Path(file.filename).name;s['source']='upload';core.save(s)
    if 'image_width' not in s:
        from PIL import Image,ImageOps
        with Image.open(s['path']) as im:
            oriented=ImageOps.exif_transpose(im);s['image_width'],s['image_height']=oriented.size
        core.save(s)
    return {'sample':s,'created':created}
@app.put('/api/samples/{sid}/review')
def review(sid:str,r:Review):
    try:s=core.sample(sid)
    except ValueError as e:raise HTTPException(404,str(e))
    if not r.reviewer.strip() or not r.group.strip() or not r.domain.strip():raise HTTPException(422,'검수자·그룹·촬영 대상은 필수입니다.')
    if r.service=='unassigned':raise HTTPException(422,'서비스를 선택하세요.')
    try:annotations.validate(r.service,r.label,r.boxes,r.annotation_complete)
    except ValueError as e:raise HTTPException(422,str(e))
    old={k:s.get(k) for k in ['service','label','reviewer','group','domain','overlay_status','capture_kind','boxes','annotation_complete']}
    s.update(boxes=[b.model_dump() for b in r.boxes],annotation_complete=r.annotation_complete,annotation_version=s.get('annotation_version',0)+1,coordinate_space='exif-oriented-normalized-xywh',overlay_status=r.overlay_status,capture_kind=r.capture_kind,service=r.service,label=r.label,reviewer=r.reviewer.strip(),group=r.group.strip(),domain=r.domain.strip(),reviewed_at=core.now())
    if r.quality_accepted and s['quality'] and s['quality']['status']=='review':s['quality'].update(status='pass',accepted_by=r.reviewer,accepted_at=core.now())
    core.save(s);core.event('review',{'id':sid,'before':old,'after':r.model_dump()});return s
@app.get('/api/samples/{sid}/image')
def image(sid:str):
    try:s=core.sample(sid)
    except ValueError as e:raise HTTPException(404,str(e))
    from PIL import Image, ImageOps
    import io
    with Image.open(s['path']) as im:
        rendered=ImageOps.exif_transpose(im).convert('RGB');rendered.thumbnail((2400,2400));buffer=io.BytesIO();rendered.save(buffer,format='PNG')
    return Response(buffer.getvalue(),media_type='image/png')
@app.get('/api/samples/{sid}/reference/{rid}')
def reference(sid:str,rid:str):
    try:s=core.sample(sid)
    except ValueError as e:raise HTTPException(404,str(e))
    ref=next((r for r in s['references'] if r['id']==rid),None)
    if not ref:raise HTTPException(404,'참조 없음')
    return FileResponse(ref['path'],media_type='application/pdf')
@app.post('/api/process/{stage}')
def process(stage:Literal['clean','features']):return submit(core.process,stage)
@app.post('/api/training')
def train(req:TrainingRequest):
    status,_=core.readiness(req.service)
    if not status['ready']:raise HTTPException(422,' '.join(status['reasons']))
    return submit(training.train,req.service)
@app.get('/api/jobs/{jid}')
def job(jid:str):
    if jid not in jobs:raise HTTPException(404,'작업이 없거나 서버 재시작으로 만료되었습니다.')
    return jobs[jid]
@app.get('/api/report')
def report():return Response(json.dumps(core.overview(),ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=yn-readiness.json'})
@app.post('/api/models/{runid}/predict/{sid}')
def predict(runid:str,sid:str):
    import joblib,numpy as np
    with core.db() as c:row=c.execute('SELECT data FROM runs WHERE id=?',(runid,)).fetchone()
    if not row:raise HTTPException(404,'등록된 모델이 없습니다.')
    run=json.loads(row['data'])
    try:s=core.sample(sid)
    except ValueError as e:raise HTTPException(404,str(e))
    if s['service']!=run['service']:raise HTTPException(422,'서비스가 다른 이미지입니다.')
    if not s['quality'] or s['quality']['status']!='pass':raise HTTPException(422,'이미지 품질 검수를 먼저 완료하세요.')
    if collection.exclusions(s):raise HTTPException(422,' '.join(collection.exclusions(s)))
    features,_,_=core.extract(s['path']);model=joblib.load(core.ROOT/'models'/run['id']/'model.joblib')
    proba=float(model.predict_proba(np.array([features]))[0,1]);result={'model':runid,'sample':sid,'ng_probability':proba,'prediction':'이상 후보' if proba>=.5 else '정상 후보','notice':'전체 이미지 기준모델 결과입니다. OCR·위치 판정 및 출하 승인을 대체하지 않습니다.'}
    core.event('prediction',result);return result

@app.get('/api/collection/template')
def collection_template():
    return Response(json.dumps(collection.template(),ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=collection-template.json'})
@app.get('/api/collection/queue')
def collection_queue():
    return Response(json.dumps(collection.queue(core.all_samples()),ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=collection-review-queue.json'})

@app.get('/api/annotations/export')
def annotation_export():
    rows=core.all_samples()
    items=[{k:s.get(k) for k in ['id','name','group','service','label','reviewer','reviewed_at','boxes','annotation_complete','annotation_version','coordinate_space','overlay_status','capture_kind']} for s in rows if s.get('annotation_complete') and s.get('reviewer') and not collection.exclusions(s)]
    return Response(json.dumps({'version':'annotation-v1','coordinate_space':'exif-oriented-normalized-xywh','class_names':annotations.DEFECTS,'eligible_count':len(items),'items':items,'notice':'검수 완료·표시 없음·촬영 도메인 일치 자료만 포함. 학습/시험 분할과 품질 점검은 별도로 필요합니다.'},ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=annotations.json'})

@app.get('/api/reference-model')
def reference_model_report():
    path=Path(__file__).resolve().parents[1]/'output'/'reference-model'/'report.json'
    return json.loads(path.read_text()) if path.exists() else None

@app.get('/api/model-benchmark')
def benchmark_report():
    path=Path(__file__).resolve().parents[1]/'output'/'model-benchmark'/'report.json'
    return json.loads(path.read_text()) if path.exists() else None

@app.get('/api/neural-benchmark')
def neural_benchmark_report():
    path=Path(__file__).resolve().parents[1]/'output'/'neural-benchmark'/'report.json'
    return json.loads(path.read_text()) if path.exists() else None

@app.get('/api/paired-cnn')
def paired_cnn_report():
    path=Path(__file__).resolve().parents[1]/'output'/'paired-cnn'/'report.json'
    return json.loads(path.read_text()) if path.exists() else None

@app.get('/api/cnn-sweep')
def cnn_sweep_report():
    path=Path(__file__).resolve().parents[1]/'output'/'cnn-hyperparameters'/'report.json'
    return json.loads(path.read_text()) if path.exists() else None

from .inspections import router as inspection_router
app.include_router(inspection_router)

web=Path(__file__).resolve().parents[1]/'web'/'dist'
if web.exists():app.mount('/',StaticFiles(directory=web,html=True),name='web')
