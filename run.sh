#!/usr/bin/env bash
# PhysiCar 자율주행 -- 이 스크립트 하나로 end2end 주행 시작.
# 새 트랙/새 카메라 각도에서 처음 돌릴 때는 config/base.py의 DRIVE_ENABLED를
# False로 두고, 웹 디버그 뷰(http://localhost:5000)로 인식이 맞는지 먼저 확인한 뒤에
# True로 바꿀 것.
set -e

cd "$(dirname "$0")"   # 어디서 실행하든 항상 이 스크립트가 있는 디렉토리 기준으로 동작

exec python3 -u main.py   # -u: 출력 버퍼링 없이 즉시 표시 (터미널/로그에 실시간으로 보이게)
