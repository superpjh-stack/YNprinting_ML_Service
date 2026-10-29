from pathlib import Path
import hashlib,json,time,platform
import joblib,numpy as np
from .neural import Autoencoder,NAMES,image_patches
from .model_benchmark import read_array,perturb,evaluate,KINDS
from .reference_training import partition
from . import core

def run(manifest='output/reference-images/manifest.json',output='output/neural-benchmark',epochs=80):
    p=Path(manifest);rows=json.loads(p.read_text())['items'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for r in rows:r['image_sha256']=hashlib.sha256((p.parent/r['image']).read_bytes()).hexdigest()
    parts=partition(rows);features=[];synthetics=[]
    for i,r in enumerate(rows):
        a=read_array(p.parent/r['image']);features.append(image_patches(a));synthetics.append({k:image_patches(perturb(a,k)) for k in KINDS} if i not in parts['train'] else {})
    train=np.concatenate([features[i] for i in parts['train']]);val=np.concatenate([features[i] for i in parts['calibration']]);results=[]
    for name in NAMES:
        t=time.perf_counter();model=Autoencoder(name).fit(train,val,epochs=epochs);fit_s=time.perf_counter()-t
        threshold=float(np.quantile([model.score(features[i]) for i in parts['calibration']],.95,method='higher'))
        validation=evaluate(model,features,synthetics,parts['calibration'],threshold);test=evaluate(model,features,synthetics,parts['test'],threshold)
        timings=[]
        for i in parts['test']:
            for _ in range(3):
                t=time.perf_counter();model.score(features[i]);timings.append((time.perf_counter()-t)*1000)
        joblib.dump({'model':model,'threshold':threshold,'feature_version':'gray-grid4-patch16-v1','production_ready':False},out/f'{name}.joblib')
        np.savez_compressed(out/f'{name}-weights.npz',**model.p)
        results.append({'name':name,'parameters':sum(v.size for v in model.p.values()),'best_epoch':model.best_epoch,'fit_seconds':fit_s,'threshold':threshold,'history':model.history,'validation':validation,'test':test,'inference_ms_p95':float(np.quantile(timings,.95))})
        print(name,'finished',round(fit_s,2),'seconds',flush=True)
    recommended=sorted(results,key=lambda r:(-(r['validation']['synthetic']['f1'] or 0),r['name']))[0]['name']
    report={'kind':'CNN/RNN/MLP 정상 시안 복원 신경망 실험','created_at':core.now(),'source_count':len(rows),'split_counts':{k:len(v) for k,v in parts.items()},'recommended_experimental_model':recommended,'models':results,'production_ready':False,
    'protocol':{'framework':'NumPy explicit backpropagation + Adam','epochs_max':epochs,'learning_rate':.001,'batch_size':32,'early_stopping':'calibration normal MSE; patience 12','grid':4,'patch_size':16,'latent_dim':16,'seed':42,'threshold':'normal calibration image score 95th percentile higher','selection':'calibration synthetic F1, alphabetical tie'},
    'limitations':['실제 불량 성능 아님; 정상 PDF 및 합성 3유형만 평가','기존 시험 시안을 재사용한 탐색 비교','RNN의 순서는 패치 내 이미지 행이며 생산 시계열이 아님','16x16 축소로 미세 문안 결함이 소실될 수 있음','프레임워크 설치 DNS 실패로 소형 NumPy 구현 사용','목형 칼날 학습은 아님','보정 세트로 조기 종료·임계값·후보 선정하므로 최종 신규 독립 평가 필요']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(out/'dataset.json').write_text(json.dumps({'items':rows,'splits':parts,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},ensure_ascii=False,indent=2));return report
if __name__=='__main__':run()
