import json
from pathlib import Path
import numpy as np,pytest,joblib
from PIL import Image
from fastapi.testclient import TestClient
from api import core,inspections
from api.paired_cnn import PairedCNN
from api.main import app

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'ROOT',tmp_path/'runtime');monkeypatch.setattr(inspections,'PROJECT',tmp_path)
    root=tmp_path/'output/reference-images';root.mkdir(parents=True)
    a=np.random.default_rng(1).integers(0,255,(128,128),dtype=np.uint8);Image.fromarray(a).save(root/'ref.png')
    (root/'manifest.json').write_text(json.dumps({'items':[{'id':'ref:1','image':'ref.png','width':128,'height':128,'group':'test'}]}))
    models=tmp_path/'output/cnn-hyperparameters';models.mkdir();joblib.dump({'model':PairedCNN(),'threshold':.5},models/'kernel5.joblib')
    s,_=core.ingest(root/'ref.png',service='plate')
    payload={'request_key':'request-123','reference_id':'ref:1','sample_id':s['id'],'reference_roi':{'x':0,'y':0},'sample_roi':{'x':0,'y':0},'operator':'tester','no_overlay_confirmed':True,'experimental_ack':True}
    return payload

def test_inspection_persistence_idempotency_and_decisions(setup):
    with TestClient(app) as c:
        assert c.get('/api/references/ref:1/preview').status_code==200
        first=c.post('/api/inspections',json=setup);assert first.status_code==200,first.text
        r=first.json();assert r['alignment']['accepted'];assert r['score'] is not None
        assert c.post('/api/inspections',json=setup).json()['id']==r['id']
        assert c.post('/api/inspections',json={**setup,'lot':'changed'}).status_code==409
        assert c.get('/api/inspections').json()['total']==1
        for decision in ['NG','RECHECK']:
            assert c.post(f"/api/inspections/{r['id']}/decisions",json={'reviewer':'qa','disposition':decision,'note':'checked'}).status_code==200
        assert len(c.get(f"/api/inspections/{r['id']}").json()['decisions'])==2
        assert c.get('/api/inspections?limit=10000').status_code==422

def test_inspection_rejects_invalid_or_unavailable(setup):
    with TestClient(app) as c:
        for fields,status in [({'service':'die'},409),({'experimental_ack':False},422),({'no_overlay_confirmed':False},422),({'sample_roi':{'x':100,'y':0}},422),({'model_id':'../../bad'},422),({'reference_id':'absent'},404)]:
            response=c.post('/api/inspections',json={**setup,**fields});assert response.status_code==status,response.text
        s=core.sample(setup['sample_id']);s['overlay_status']='present';core.save(s)
        assert c.post('/api/inspections',json=setup).status_code==422

def test_alignment_failure_is_saved_inconclusive(setup,tmp_path):
    p=tmp_path/'different.png';Image.fromarray(np.random.default_rng(77).integers(0,255,(128,128),dtype=np.uint8)).save(p);s,_=core.ingest(p)
    with TestClient(app) as c:
        r=c.post('/api/inspections',json={**setup,'sample_id':s['id']}).json()
        assert r['status']=='inconclusive' and r['score'] is None
        assert c.get('/api/inspections').json()['total']==1
