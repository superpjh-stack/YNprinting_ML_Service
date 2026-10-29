# 와이엔프린팅 ML 작업실

두 비전 서비스의 데이터 준비와 전체 이미지 OK/NG 기준모델을 위한 로컬 개발 버전입니다. 실제 OCR 문자 비교·객체 위치 검출 모델은 아직 구현/연결되지 않았으며, 이 버전을 완성된 문안/칼날 검사기로 사용하지 않습니다.

## 현재 구현

- 이미지 원본 SHA-256 보존, 폴더/Excel 약한 라벨 수집, 같은 폴더의 PDF 참조 후보 연결
- 회전 메타데이터 처리, 해상도/노출/흐림 품질 점검, 22개 이미지 피처
- 이미지·PDF 확인, 서비스/그룹/정답/검수자/촬영 대상 등록, 변경 감사 로그
- 정상·불량 및 그룹 조건을 검사하고 미검수 데이터 학습 차단
- 동일 참조 또는 그룹의 분할 누수 방지, 다수 클래스·로지스틱·랜덤 포레스트 비교
- validation 기반 모델 선택, test 평가, 모델/데이터 스냅샷/평가 파일 저장, 예측 API
- 한국어 React 화면: 수집 → 정제 → 피처 → 학습 → 성능측정

## 실행

현재 작업공간은 Python `.venv`, 프런트엔드 `web/node_modules`, 빌드 `web/dist`를 구성했습니다. 외부 다운로드 DNS 실패로 기존 로컬 패키지를 복사했으며 다른 프로젝트의 파일은 변경하지 않았습니다. `requirements-tested.txt`와 `web/package.json`은 실제 검증한 버전을 기록합니다.

```sh
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8042
```

브라우저에서 http://127.0.0.1:8042 를 엽니다. API 문서는 `/docs`입니다. 현재 자동화 환경의 포트 바인딩 제한으로 서버를 계속 실행해 두지는 못했습니다. 사용자의 일반 터미널에서 위 명령으로 실행하세요.

새 환경 설치:

```sh
uv venv
uv pip install -r requirements-tested.txt
npm install --prefix web
npm run build --prefix web
```

CLI:

```sh
.venv/bin/python -m api.cli import --source '/원본/폴더'
.venv/bin/python -m api.cli clean
.venv/bin/python -m api.cli features
.venv/bin/python -m api.cli report
.venv/bin/python -m api.cli train --service plate
.venv/bin/python -m api.cli train --service die
```

데이터는 `.runtime`에 보존합니다. 환경변수 `YN_RUNTIME`으로 변경할 수 있습니다. 원본 경로를 받는 가져오기는 관리자 CLI 전용입니다. 화면은 PNG/JPG 업로드를 지원합니다. 폴더 재수집 시 같은 해시는 중복 추가하지 않습니다.

## 실제 자료 처리 결과

267개 이미지, 60개 폴더 그룹을 수집했습니다. 267개 모두 자동 품질 점검 및 피처 추출에 성공했고 232개에 같은 폴더의 PDF 참조 후보가 있습니다. 35개는 PDF 후보가 없습니다. 자동 품질 통과는 문안 판독·검사 적합성의 승인과 다릅니다.

폴더별 불량 사유는 267개에 연결되었으나 이미지별 정답이 아니므로 모두 미검수로 유지합니다. 서비스 분류도 사람이 확인해야 합니다. 따라서 **실제 학습 실행 0건, 실제 성능 측정값 없음**입니다. 테스트의 합성 데이터는 임시 격리 환경에서만 학습하며 현장 성능으로 보고하지 않습니다.

## 다음 데이터 요건

문안: 정상 인쇄판 실촬영본, 이상 촬영본, 대응 시안/개정판, 문자 정답 및 결함 위치/유형.
목형: 정상/이상 목형 사진과 확대 사진, 기준 형상, 목형 ID, 결함 위치/유형 및 검수자.
같은 시안·목형·LOT 및 관련 crop은 동일 그룹으로 관리해야 합니다. 파일명만 보고 라벨을 확정하지 않습니다.

## 검증과 제한

```sh
.venv/bin/python -m pytest -q
npm run build --prefix web
```

현재 기준모델은 개별 문구나 칼날 결함 위치를 설명하지 못합니다. 학습 최소 12장/6그룹 조건은 개발 실행 조건일 뿐 성능 보장 표본수가 아닙니다. 그룹 분할의 실제 비율은 약 60/20/20이며 각 분할에 정상·불량이 모두 있어야 합니다. 충분한 데이터가 없으면 실패 사유를 반환합니다.

OCR/YOLO/Autoencoder, 좌표 라벨 편집, 이미지 정합, 현장 CPU 지연 측정, 신뢰구간, 배포·인증·MES·카메라 연동은 후속 구현 항목입니다. 현재 작업 상태는 메모리에 있어 서버 재시작 시 진행 상태 조회가 만료됩니다. 학습 완료 결과와 원본·검수 이력은 DB/파일에 남습니다. 동시 여러 사용자 운영이 아닌 loopback 단일 사용자 개발 환경입니다.

데이터 분할 근거: [scikit-learn GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [Pipeline 및 데이터 누수 방지](https://scikit-learn.org/stable/common_pitfalls.html).

## 촬영 및 표시 검수 추가

`docs/collection/capture-guide.md`에 현장 촬영 순서와 서비스별 추가 수집 요건을 정리했습니다. `collection-template.json`은 촬영 기록 양식, `review-queue.json`은 기존 267장 검수 대기 목록입니다. 화면에서도 두 JSON을 내려받을 수 있습니다. JSON 일괄 입력은 아직 지원하지 않습니다.

검수 화면에서 `촬영 대상 분류`와 `결함 위치 표시·필기`를 확인합니다. 표시 미확인/있음 및 서비스와 다른 촬영 도메인은 학습과 예측에서 제외됩니다. 자동 품질 점검 통과로 이 검수를 대체할 수 없습니다. 기존 기록의 누락값은 미확인으로 처리하며 정답이나 서비스 분류를 자동 확정하지 않습니다.

## 결함 위치 라벨링

이미지 정답 검수 창에서 결함 유형을 선택하고 사진 위를 드래그하거나 `좌표로 영역 추가` 버튼을 사용합니다. 좌표는 EXIF 회전을 반영한 이미지의 0~1 정규화 xywh입니다. 위치·유형을 확인한 뒤 `결함 위치·유형 검수 완료`를 체크하고 검수 결과를 저장합니다. 원본 파일에는 표시를 그리지 않습니다.

문안 서비스에는 문자 누락/오탈자/번짐/점·기호/인쇄 밀림, 칼날 서비스에는 마모/휨/파손/위치 불량 유형을 제공합니다. 문안의 기준 문구와 실제 관찰 문구도 수동 기록할 수 있습니다. OCR 자동 판독 기능은 아닙니다.

`검수 라벨 내보내기`는 정답 검수와 위치 검수를 완료하고, 표시 없음 및 서비스 촬영 대상 일치가 확인된 자료만 JSON으로 제공합니다. 모델 학습 전 품질 검사와 그룹 분할은 여전히 별도로 필요합니다. 현재 기준모델은 이미지 OK/NG 분류기이며, 이번에 저장한 위치 라벨을 사용한 검출 모델 학습은 아직 연결되지 않았습니다.

## 인공신경망 실험

CNN-AE, RNN-AE, MLP-AE를 추가하고 정상 시안으로 학습했습니다. `output/neural-benchmark/README.md`에서 실제 결과와 한계를 확인할 수 있습니다. PyTorch 설치가 DNS 오류로 실패하여 NumPy 역전파/Adam 구현으로 수행했습니다. 실행 명령은 `.venv/bin/python -m api.neural_benchmark`입니다. 합성 시험 Recall/F1은 모두 0으로, 모델을 운영용으로 승인하지 않았습니다.

## CNN 하이퍼파라미터 5종 추가

`output/cnn-hyperparameters/README.md`에서 기존 모델과 채널12·채널24·커널5·학습률0.001·60epoch 설정을 비교합니다. 동일 합성 쌍에서 검증 F1로 선택한 실험 후보는 kernel5입니다. 실제 사진 검증 또는 운영 승인은 아닙니다. 실행은 `.venv/bin/python -m api.cnn_sweep`입니다.

## 검사 애플리케이션

앱 첫 화면 `검사·이력`에서 기준 시안 선택 → 사진 등록 → 대응 영역 지정 → CNN 검사 → 작업자 판정 저장을 사용할 수 있습니다. 실행: `./scripts/start-local.sh`. 상세 사용법 및 상용 운영 전 미구현 항목은 `docs/service-pilot.md`에 정리했습니다. 현재는 인증 없는 loopback 단일 사용자 로컬 파일럿이며 원격 서버 배포는 하지 않았습니다.
