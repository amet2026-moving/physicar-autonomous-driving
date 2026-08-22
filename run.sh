#!/usr/bin/env bash
# PhysiCar 자율주행 -- 이 스크립트 하나로 end2end 주행이 시작됩니다.
# 최초 실행 전 config.py의 DRIVE_ENABLED가 False인지 확인하세요 (기본값 False) --
# 디버그 웹 뷰(http://localhost:5000)에서 차선 인식이 맞는지 본 뒤에만 True로 바꾸세요.
set -e

cd "$(dirname "$0")"

PYTHON=python3
command -v python3 >/dev/null 2>&1 || PYTHON=python

if [ ! -d .venv ]; then
    echo "[run.sh] .venv 없음 -- 새로 생성합니다"
    "$PYTHON" -m venv .venv
fi

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
fi

pip install -q -r requirements.txt

exec python main.py
