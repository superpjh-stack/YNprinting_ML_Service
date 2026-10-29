import hashlib,json,uuid
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from . import core

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel().tolist()
    div=lambda a,b:a/b if b else None
    return dict(n=len(y),accuracy=float(accuracy_score(y,p)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),confusion_matrix=[[tn,fp],[fn,tp]])

def split(rows):
    # Merge groups sharing a reference, preventing copies of a design crossing splits.
    parent={s['group']:s['group'] for s in rows}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    refs={}
    for row in rows:
        for ref in row.get('references',[]):
            rid=ref['id']
            if rid in refs:parent[find(row['group'])]=find(refs[rid])
            else:refs[rid]=row['group']
    groups=np.array([find(s['group']) for s in rows]);y=np.array([int(s['label']=='NG') for s in rows]);idx=np.arange(len(rows))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(idx,y,groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=43).split(trainval,y[trainval],groups[trainval]))
    train,val=trainval[tr],trainval[va]
    if any(len(set(y[a]))<2 for a in [train,val,test]):raise ValueError('고정 그룹 분할의 train/validation/test에 정상·불량이 모두 필요합니다. 표본을 보강하세요.')
    return train,val,test

def train(service):
    status,rows=core.readiness(service)
    if not status['ready']:raise ValueError(' '.join(status['reasons']))
    tr,va,te=split(rows);X=np.array([r['features']['values'] for r in rows]);y=np.array([int(r['label']=='NG') for r in rows])
    snapshot=[{k:r.get(k) for k in ['id','group','label','reviewer','reviewed_at','features','references','domain','overlay_status','capture_kind']} for r in rows]
    digest=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
    runid=service+'-'+digest[:20]
    with core.db() as c:
        existing=c.execute('SELECT data FROM runs WHERE id=?',(runid,)).fetchone()
        if existing:return json.loads(existing['data'])
    models={'majority':DummyClassifier(strategy='most_frequent'),'logistic':make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',random_state=42,max_iter=2000)),
      'random_forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=1)}
    comparisons=[]
    for name,m in models.items():
        m.fit(X[tr],y[tr]);score=metrics(y[va],m.predict(X[va]));comparisons.append({'name':name,'validation':score})
    eligible=[c for c in comparisons if c['name']!='majority' and (c['validation']['recall'] or 0)>=.85]
    candidates=eligible or [c for c in comparisons if c['name']!='majority']
    champion=max(candidates,key=lambda c:c['validation']['f1'] or 0)['name']
    chosen=models[champion];testpred=chosen.predict(X[te]);folder=core.ROOT/'models'/runid;folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(chosen,folder/'model.joblib')
    report={'id':runid,'service':service,'created_at':core.now(),'kind':'전체 이미지 OK/NG 기준모델','feature_version':'gray-edge-v1','dataset_sha256':digest,
      'status':'validation_recall_met' if eligible else 'candidate_only','champion':champion,'comparisons':comparisons,'test':metrics(y[te],testpred),
      'split':{name:[rows[i]['id'] for i in idx] for name,idx in [('train',tr),('validation',va),('test',te)]},
      'test_predictions':[{'id':rows[i]['id'],'actual':int(y[i]),'predicted':int(p)} for i,p in zip(te,testpred)],
      'seed':42,'threshold':.5,'limitations':['OCR·위치 검출 성능이 아닙니다.','계약 성능 검수 미승인','소규모 기준모델; 현장 독립 평가 필요']}
    (folder/'dataset.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2));(folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with core.db() as c:c.execute('INSERT INTO runs VALUES (?,?)',(runid,json.dumps(report,ensure_ascii=False)))
    core.event('training',{'run_id':runid});return report
