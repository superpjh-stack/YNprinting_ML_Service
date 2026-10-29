#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo 'Python 환경이 없습니다. README의 설치 절차를 실행하세요.' >&2
  exit 1
fi
if [ ! -f web/dist/index.html ]; then
  npm run build --prefix web
fi
exec .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port "${YN_PORT:-8042}"
