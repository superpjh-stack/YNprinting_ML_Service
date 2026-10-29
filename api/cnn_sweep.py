"""Prespecified one-factor CNN sweep; test is never used for selection."""
from pathlib import Path
import hashlib,json,time
import numpy as np,joblib
from sklearn.metrics import roc_auc_score
from .paired_cnn import PairedCNN
from .paired_experiment import build
from .reference_training import partition
from .training import metrics
from . import core
BASE={'channels':6,'kernel':3,'learning_rate':.003,'batch_size':16,'epochs':30}
CONFIGS=[('baseline',BASE),('channels12',{**BASE,'channels':12}),('channels24',{**BASE,'channels':24}),('kernel5',{**BASE,'kernel':5}),('lr0001',{**BASE,'learning_rate':.001}),('epochs60',{**BASE,'epochs':60})]

def choose_threshold(y,s):
    choices=[]
    for t in np.unique(np.r_[0,s,1]):
        pred=s>t;fpr=float(pred[y==0].mean())
        if fpr<=.05:choices.append((metrics(y,pred)['f1'] or 0,float(t)))
    return max(choices)[1]

def evaluate(y,s,t):
    result=metrics(y,s>t);result.update(auroc=float(roc_auc_score(y,s)),normal_alarm_rate=float((s[y==0]>t).mean()));return result

def run():
    out=Path('output/cnn-hyperparameters');out.mkdir(parents=True,exist_ok=True)
    p=Path('output/reference-images/manifest.json');rows=json.loads(p.read_text())['items']
    for r in rows:r['image_sha256']=hashlib.sha256((p.parent/r['image']).read_bytes()).hexdigest()
    parts=partition(rows);sets,failed=build(rows,p.parent,parts);tr,va,te=[sets[k] for k in ['train','calibration','test']]
    datahash=hashlib.sha256(b''.join(s[k].tobytes() for s in [tr,va,te] for k in ['x','y'])).hexdigest()
    results=[];models={}
    for name,cfg in CONFIGS:
        start=time.perf_counter();m=PairedCNN(channels=cfg['channels'],kernel=cfg['kernel']).fit(tr['x'],tr['y'],va['x'],va['y'],epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],batch_size=cfg['batch_size']);elapsed=time.perf_counter()-start
        vs=m.forward(va['x'])[0];t=choose_threshold(va['y'],vs)
        result={'name':name,'config':cfg,'best_epoch':m.best_epoch,'parameter_count':sum(a.size for a in m.p.values()),'training_seconds':elapsed,'threshold':t,'validation':evaluate(va['y'],vs,t),'history':m.history}
        results.append(result);models[name]=m
        joblib.dump({'model':m,'threshold':t,'config':cfg,'data_sha256':datahash,'production_ready':False},out/f'{name}.joblib');np.savez_compressed(out/f'{name}-weights.npz',**m.p)
        print(name,'trained',round(elapsed,2),'seconds',flush=True)
    # Freeze choice before reading test predictions.
    winner=sorted(results,key=lambda r:(-(r['validation']['f1'] or 0),r['parameter_count'],r['name']))[0]['name']
    (out/'selection.json').write_text(json.dumps({'candidate':winner,'rule':'validation F1 desc, parameter count asc, name asc','data_sha256':datahash},indent=2))
    for r in results:
        ps=models[r['name']].forward(te['x'])[0];r['test']=evaluate(te['y'],ps,r['threshold']);r['test_scores']=ps.tolist()
    report={'kind':'대응 영역 CNN 하이퍼파라미터 추가 5종','created_at':core.now(),'data_sha256':datahash,'split_counts':{k:len(v) for k,v in parts.items()},'pair_counts':{k:len(s['y']) for k,s in sets.items()},'excluded_pairs':len(failed),'candidate':winner,'selection_rule':'validation F1 desc, parameter count asc, name asc','threshold_rule':'validation F1 max under normal alarm <= 5%; tie higher threshold','production_ready':False,'models':results,'limitations':['정합 통과한 합성 ROI 쌍 평가이며 실제 사진 성능 아님','기존 시험 시안 반복 사용: 탐색 비교, 최종 독립 시험 아님','단일 초기화 시드 42; 여러 시드 반복 검증 전','같은 원본 파생 ROI는 독립 표본 아님','운영 후보 승인은 하지 않음']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(out/'dataset.json').write_text(json.dumps({'source_rows':rows,'splits':parts,'pair_metadata':{k:s['meta'] for k,s in sets.items()},'test_labels':te['y'].tolist(),'excluded':failed},ensure_ascii=False,indent=2))
    return report
if __name__=='__main__':
    r=run();print(json.dumps({'candidate':r['candidate'],'models':[{k:m[k] for k in ['name','config','validation','test']} for m in r['models']]},indent=2))
