# physicar-autonomous-driving

PhysiCar 자율주행 — 2026 AMET 자율주행 해커톤용 프로젝트입니다.

팀원별로 병렬 개발되던 여러 버전의 검증된 로직을 하나의 협업 구조로 통합했습니다! 

- 라인트레이싱
- 장애물 회피
- 신호등 인식
- 주행 설정값

각 기능은 버전 파일로 관리하며, 실제 실행에 사용할 조합은 versions.py에서 선택합니다.

## 실행

```bash
./run.sh
```

`python3 -u main.py`를 바로 실행합니다. PhysiCar 컨테이너엔 필요한 패키지(numpy/opencv 등)가
이미 시스템에 깔려 있어서 별도 venv/pip 설치 단계가 없습니다 -- 로컬에서 처음 돌려보는
환경이라면 `requirements.txt`를 미리 `pip install -r requirements.txt`로 설치해두세요.

## 구조

```
moving/
├── run.sh                 # 실행 진입점
├── main.py                # 전체 기능 연결 + 메인 주행 루프
├── versions.py            # 실행할 버전 조합 선택 + 주행 기록
├── car_api.py             # PhysiCar 카메라·LiDAR·주행 API 래퍼
├── debug_view.py          # Flask 디버그 웹 뷰
│
├── config/                # 전체 튜닝값 버전
│   ├── v1_basic.py
│   └── ...
│
├── lane/                  # 라인트레이싱 버전
│   ├── v1_basic.py
│   └── ...
│
├── obstacle/              # 장애물 회피 버전
│   ├── v1_basic.py
│   └── ...
│
├── traffic_light/         # 신호등 인식 버전
│   ├── v1_basic.py
│   └── ...
│
├── requirements.txt
└── README.md

```
각 폴더의 역할은 다음과 같습니다.

- config/ : 속도, 거리, gain, threshold 등 전체 튜닝값
- lane/ : 차선 인식, 코너 처리, 조향 및 속도 제어
- obstacle/ : LiDAR 기반 장애물 회피 및 후진 탈출
- traffic_light/ : 신호등 탐색, 인식 및 출발 처리
- versions.py : 위 버전들을 조합하여 실제 실행 버전 결정

## 버전 관리 & 주행 기록 (`versions.py`)

이 프로젝트는 버전을 브랜치가 아니라 **파일**로 관리합니다. 뭔가 새로 시도할 때마다
`lane/`, `obstacle/`, `traffic_light/`, `config/` 폴더 안에 새 버전 파일을 추가하고,
`versions.py`에서 어떤 조합을 쓸지 고르는 방식입니다. 기존 파일은 그대로 남아있으니
"이전 버전으로 되돌리기"도 `versions.py`에서 주석 on/off만하면 됩니다.

### 새 버전 추가하는 법 (예시)

장애물 회피를 더 공격적으로 튜닝해보고 싶다면:

**1. 기존 버전 파일을 복사해서 새 버전을 만듭니다.**
```bash
cp obstacle/v1_basic.py obstacle/v2_aggressive.py
# v2_aggressive.py 안에서 원하는 부분만 수정
```

**2. `versions.py` 맨 아래에 새 블록을 추가합니다.** (이전 활성 블록은 `Ctrl+/`로 주석 처리)
```python
"""0824 다현
설명: OBSTACLE_STEER_GAIN 55->70으로 올려서 회피 반응을 더 크게 함
점수: 58초
충돌: 0회
이탈: 0회
"""
CONFIG_VERSION = "v1_basic"
LANE_VERSION = "v1_basic"
OBSTACLE_VERSION = "v2_aggressive"
TRAFFIC_VERSION = "v1_basic"
```
각 버전은 서로 독립적으로 조합할 수 있습니다.
실제로 활성화된 버전 설정 블록은 반드시 하나만 유지합니다.
이전 실험 블록을 남겨둘 때는 변수 선언까지 주석 처리합니다.

```
# CONFIG_VERSION = "v1_basic" 
# LANE_VERSION = "v1_basic" 
# OBSTACLE_VERSION = "v1_basic" 
# TRAFFIC_VERSION = "v1_basic"
```

**3. `./run.sh`로 테스트합니다.**

### 설명은 반드시 채우세요

`설명` 필드를 비워두면 나중에 이 조합이 뭘 바꾼 건지 아무도 모릅니다. 무엇을 왜 바꿨는지
한 줄이라도 꼭 적으세요. 실패한 시도도(랩타임이
나빠졌거나 충돌/이탈이 늘었어도) 지우지 말고 주석 처리된 블록으로 남겨두세요 -- 다른
팀원이 같은 시도를 또 반복하지 않게 해줍니다.

### 작업이 끝나면 반드시 git에 올리세요

테스트가 끝났으면 시뮬레이션을 종료하기 전에 커밋하고 푸시하세요. 이 프로젝트에서 Git은 복잡한 브랜치 관리보다는 서버의 최신 코드를 GitHub에 백업하고, 팀원들이 로컬에서 빠르게 확인하거나 내려받기 위한 용도​로 사용합니다.

서버에는 GitHub 인증이 미리 설정되어 있으므로, 작업이 끝나면 아래 순서대로 업로드합니다.

```bash
git add . 
git commit -m "[작업자] 변경 내용" 
git push
```


## 로컬 노트북 <-> 시뮬레이터 작업 방법

### 방법 1. 가장 쉬운 방법 — 추천
GitHub에서 코드를 로컬로 내려받아 확인하거나 수정합니다.

처음 한 번:
git clone https://github.com/amet2026-moving/physicar-autonomous-driving.git . 

이후 최신 코드 받기:
git pull

로컬에서 수정한 파일이 있다면 해당 파일만 서버의 moving/ 폴더로 직접 옮깁니다.
이후 최종 실행 테스트와 GitHub 업로드는 서버에서 하는 것을 기본으로 합니다.

### 방법 2. Git에 익숙한 경우
각자 로컬 PC에서 자신의 GitHub 계정을 인증한 뒤 직접 push해도 됩니다.
```bash
git pull 
# 작업 이후
git add . 
git commit -m "[이름] 변경 내용" 
git push
```

## 팀 작업 규칙
1. 작업 시작 전 git pull
2. 기존 버전을 바로 덮어쓰기보다 새 버전 파일 추가
3. 버전명은 v번호_특징.py 형식 권장
4. 테스트 결과는 versions.py에 기록
5. 실패한 실험도 가능하면 기록 유지
6. 같은 파일을 여러 명이 동시에 수정하지 않기
7. 작업 완료 후 commit + push
8. 커밋 메시지에 실제 작업자 이름 작성
9. 실제 대회 실행은 항상 ./run.sh

예시 버전명:
v1_basic.py
v2_highspeed.py
v3_fast_corner.py
v4_gap_reverse.py

## 참고

- `debug_view.py`의 웹 뷰는 `assets/line-tracing/webui.html`을 찾습니다 (이 폴더 기준
  한 단계 위). 다른 위치로 배포할 때는 이 경로 가정을 확인하세요 -- 없어도 주행 자체는
  정상 동작하고 디버그 패널만 생략됩니다.
