import json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api import core,training
from api.main import app

@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'ROOT',tmp_path/'runtime')

def make_image(tmp_path,name='sample.png',seed=1):
    p=tmp_path/name;Image.fromarray(np.random.default_rng(seed).integers(0,255,(160,180,3),dtype=np.uint8)).save(p);return p

def test_raw_duplicate_and_features(tmp_path):
    p=make_image(tmp_path);s,created=core.ingest(p);assert created
    assert not core.ingest(p)[1]
    core.process('clean');core.process('features');s=core.sample(s['id'])
    assert s['quality']['status']=='pass';assert len(s['features']['values'])==22
    assert p.read_bytes()==open(s['path'],'rb').read()

def test_weak_labels_do_not_allow_training(tmp_path):
    core.ingest(make_image(tmp_path),service='plate',weak_label='불량')
    core.process('clean');core.process('features')
    assert not core.readiness('plate')[0]['ready']
    with pytest.raises(ValueError):training.train('plate')
    with TestClient(app) as c:
        assert c.post('/api/training',json={'service':'plate'}).status_code==422
        assert c.get('/api/samples?limit=10000').status_code==422

def test_review_audit_and_missing_models(tmp_path):
    s,_=core.ingest(make_image(tmp_path));core.process('clean')
    with TestClient(app) as c:
        payload={'service':'plate','label':'NG','reviewer':'tester','group':'artwork-A','domain':'plate'}
        assert c.put(f"/api/samples/{s['id']}/review",json=payload).status_code==200
        assert core.sample(s['id'])['group']=='artwork-A'
        assert c.post(f"/api/models/missing/predict/{s['id']}").status_code==404
        payload['reviewer']=' '
        assert c.put(f"/api/samples/{s['id']}/review",json=payload).status_code==422
    with core.db() as db:assert db.execute("SELECT count(*) FROM events WHERE kind='review'").fetchone()[0]==1

def test_training_group_integrity_and_persistence(tmp_path):
    # Synthetic fixture exercises mechanics only; never imported into real runtime.
    for g in range(20):
        for label in [0,1]:
            sid=f'fixture-{g}-{label}'
            core.save(dict(id=sid,service='plate',group=f'g{g}',label='NG' if label else 'OK',reviewer='test',reviewed_at='fixture',overlay_status='absent',capture_kind='plate_capture',
              quality={'status':'pass'},features={'values':[float(label),float(g)/20]+[0.]*20}))
    rows=core.all_samples();tr,va,te=training.split(rows)
    groups=[{rows[i]['group'] for i in indices} for indices in [tr,va,te]]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    report=training.train('plate');assert report['test']['n']>0
    assert (core.ROOT/'models'/report['id']/'model.joblib').exists()
    assert training.train('plate')['id']==report['id']
    assert report['champion'] in ['logistic','random_forest']

def test_undefined_precision():
    m=training.metrics([0,1],[0,0]);assert m['precision'] is None;assert m['recall']==0

def test_invalid_upload():
    with TestClient(app) as c:
        assert c.post('/api/uploads',files={'file':('x.png',b'bad','image/png')}).status_code==422
        assert c.post('/api/uploads',files={'file':('x.html',b'bad','text/html')}).status_code==422
        assert c.get('/api/samples/absent/image').status_code==404

def test_reference_groups_do_not_cross_splits():
    rows=[]
    for g in range(20):
        for label in [0,1]:
            rows.append({'group':f'g{g}', 'label':'NG' if label else 'OK', 'references':[{'id':f'ref{g//2}'}]})
    parts=training.split(rows)
    references=[{r['references'][0]['id'] for i,r in enumerate(rows) if i in part} for part in parts]
    assert not references[0]&references[1] and not references[0]&references[2] and not references[1]&references[2]

def test_prediction_uses_saved_preprocessing(tmp_path):
    for g in range(20):
        for label in [0,1]:
            core.save(dict(id=f'fixture-{g}-{label}',service='die',group=f'g{g}',label='NG' if label else 'OK',reviewer='test',reviewed_at='fixture',overlay_status='absent',capture_kind='die_capture',quality={'status':'pass'},features={'values':[float(label),float(g)/20]+[0.]*20}))
    report=training.train('die')
    s,_=core.ingest(make_image(tmp_path),service='die');s['quality']={'status':'pass'};s['overlay_status']='absent';s['capture_kind']='die_capture';core.save(s)
    with TestClient(app) as c:
        response=c.post(f"/api/models/{report['id']}/predict/{s['id']}")
        assert response.status_code==200
        assert 0 <= response.json()['ng_probability'] <= 1
        assert c.get('/api/health').json()['model_ready'] is True
        assert c.get('/').status_code==200

@pytest.mark.parametrize('overlay,kind', [('unknown','plate_capture'),('present','plate_capture'),('absent','label_capture'),('absent','artwork'),('absent','die_capture')])
def test_overlay_and_wrong_domain_excluded(overlay,kind):
    core.save(dict(id='guard',name='guard.png',service='plate',group='g',label='NG',reviewer='test',features={'values':[0.]*22},quality={'status':'pass'},overlay_status=overlay,capture_kind=kind))
    status,eligible=core.readiness('plate')
    assert eligible==[]
    assert any('촬영 대상' in r for r in status['reasons'])

def test_collection_download_and_review_fields(tmp_path):
    s,_=core.ingest(make_image(tmp_path))
    with TestClient(app) as c:
        queue=c.get('/api/collection/queue').json()
        assert queue['total']==1 and queue['items'][0]['overlay_status']=='unknown'
        assert len(c.get('/api/collection/template').json()['records'])==2
        payload={'service':'plate','label':'NG','reviewer':'검수자','group':'g','domain':'인쇄판','overlay_status':'present','capture_kind':'plate_capture'}
        result=c.put(f"/api/samples/{s['id']}/review",json=payload)
        assert result.status_code==200 and result.json()['overlay_status']=='present'
        payload['overlay_status']='invented'
        assert c.put(f"/api/samples/{s['id']}/review",json=payload).status_code==422

def test_annotation_validation_export_and_history(tmp_path):
    s,_=core.ingest(make_image(tmp_path))
    box=dict(defect_type='missing_text',x=.1,y=.2,width=.3,height=.4,expected_text='ABC',observed_text='AC')
    payload=dict(service='plate',label='NG',reviewer='tester',group='g',domain='plate',capture_kind='plate_capture',overlay_status='absent',annotation_complete=True,boxes=[box])
    with TestClient(app) as c:
        url=f"/api/samples/{s['id']}/review"
        assert c.put(url,json=payload).status_code==200
        exported=c.get('/api/annotations/export').json()
        assert exported['eligible_count']==1
        assert exported['items'][0]['boxes'][0]['expected_text']=='ABC'
        assert exported['items'][0]['annotation_version']==1
        assert c.put(url,json={**payload,'label':'OK'}).status_code==422
        assert c.put(url,json={**payload,'boxes':[]}).status_code==422
        assert c.put(url,json={**payload,'boxes':[{**box,'x':.9}]}).status_code==422
        assert c.put(url,json={**payload,'boxes':[{**box,'defect_type':'break'}]}).status_code==422
        assert c.put(url,json={**payload,'overlay_status':'present'}).status_code==200
        assert c.get('/api/annotations/export').json()['eligible_count']==0
    with core.db() as db:
        events=[json.loads(r['data']) for r in db.execute("SELECT data FROM events WHERE kind='review'")]
    assert events[-1]['before']['boxes'][0]['expected_text']=='ABC'

def test_oriented_preview_does_not_modify_raw(tmp_path):
    p=tmp_path/'rotated.jpg';im=Image.new('RGB',(120,80));exif=im.getexif();exif[274]=6;im.save(p,exif=exif)
    s,_=core.ingest(p);before=p.read_bytes()
    import io
    with TestClient(app) as c:
        preview=Image.open(io.BytesIO(c.get(f"/api/samples/{s['id']}/image").content))
        assert preview.size==(80,120)
    assert p.read_bytes()==before
