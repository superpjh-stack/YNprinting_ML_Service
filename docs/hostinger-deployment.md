# Hostinger VPS 배포

프로젝트 이름은 `ynprinting-ml`, 앱 포트는 8042이다. `compose.yaml`은 인증이 없는 현재 파일럿을 호스트 loopback에만 연결한다. 외부 접속에는 HTTPS와 인증을 제공하는 별도 프록시가 필요하다.

```sh
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:8042/api/health
```

Hostinger Docker Manager YAML 편집기에서는 `build: .`을 다음과 같이 바꾸어 커밋이 고정된 Git 소스를 지정한다.

```yaml
build:
  context: https://github.com/superpjh-stack/YNprinting_ML_Service.git#<commit-sha>
```

이 VPS의 Docker Manager는 초기 배포 시 이미지를 자동 빌드하지 않아 `No such image: ynprinting-ml-app:latest`로 실패한다. YAML 저장 후 웹 콘솔에서 다음 명령으로 먼저 빌드하고 시작한다. 이후 Docker Manager에서 상태와 로그를 관리한다.

```sh
cd /docker/ynprinting-ml
docker compose --progress plain build
docker compose up -d
```

새 버전을 배포할 때도 Git 커밋을 바꾼 뒤 위 빌드·시작 명령을 실행한다.

`runtime` 볼륨에는 SQLite와 업로드 파일이 저장된다. `references` 볼륨은 승인 시안용이며 읽기 전용으로 마운트된다. 초기 배포에는 고객 시안과 로컬 DB를 전송하지 않으므로 기준 목록과 검사 이력은 비어 있다. 시안이 준비되기 전에는 실제 검사를 실행할 수 없다. 실험 CNN 모델 6개는 이미지에 포함된다.

백업은 앱을 정지한 상태에서 두 볼륨을 보관한다. `docker compose down -v`는 데이터를 삭제하므로 사용하지 않는다. 롤백은 이전 커밋으로 빌드 대상을 되돌려 재배포하며 볼륨은 유지한다.

SSH 접근 권한이 있는 사용자는 다음 터널로 웹앱을 확인할 수 있다.

```sh
ssh -L 8042:127.0.0.1:8042 root@187.52.127.215
```

터널 연결 후 로컬 브라우저에서 `http://127.0.0.1:8042`를 연다.
