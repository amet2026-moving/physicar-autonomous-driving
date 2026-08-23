# physicar-autonomous-driving

PhysiCar 자율주행 -- 2026 AMET 자율주행 해커톤용. 팀원별로 병렬 개발되던 여러 버전(재모
modi_test_v5/v6, Sangheon test_v7/v8, 성찬 auto3/auto4)의 검증된 로직을 하나의 협업 구조로
합친 구현입니다.

## 실행

```bash
./run.sh
```

`python3 -u main.py`를 바로 실행합니다. PhysiCar 컨테이너엔 필요한 패키지(numpy/opencv 등)가
이미 시스템에 깔려 있어서 별도 venv/pip 설치 단계가 없습니다 -- 로컬에서 처음 돌려보는
환경이라면 `requirements.txt`를 미리 `pip install -r requirements.txt`로 설치해두세요.

## 구조

라인트레이싱 / 장애물회피 / 신호등 / 설정값, 이 4가지 서브시스템은 각각 독립된 폴더 아래
버전 파일로 나뉘어 있습니다. 어떤 버전 조합을 쓸지는 `versions.py` 한 곳에서 고릅니다.

```
moving/
├── run.sh          # 실행 진입점
├── main.py         # 전체 조립 + 메인 루프
├── versions.py     # 어떤 버전 조합을 쓸지 여기서 선택 (+ 주행 기록)
├── car_api.py      # PhysiCar 하드웨어 API 래퍼
├── debug_view.py   # Flask 디버그 웹 뷰
│
├── config/
│   └── v1_basic.py         # 전 서브시스템 튜닝 상수
├── lane/
│   └── v1_basic.py         # 차선 인식 + LaneKeeper (코너모드/조향/속도 컨트롤러)
├── obstacle/
│   └── v1_basic.py         # ObstacleAvoider -- LiDAR 기반 회피 + 후진 탈출
└── traffic_light/
    └── v1_basic.py         # 신호등 인식 및 출발 처리
```

## 버전 관리 & 주행 기록 (`versions.py`)

이 프로젝트는 버전을 브랜치가 아니라 **파일**로 관리합니다. 뭔가 새로 시도할 때마다
`lane/`, `obstacle/`, `traffic_light/`, `config/` 폴더 안에 새 버전 파일을 추가하고,
`versions.py`에서 어떤 조합을 쓸지 고르는 방식입니다. 기존 파일은 그대로 남아있으니
"이전 버전으로 되돌리기"도 `versions.py` 한 줄만 바꾸면 됩니다.

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
네 값은 서로 독립적으로 섞어 써도 됩니다 -- 위 예시처럼 `OBSTACLE_VERSION`만 바꾸고
나머지는 그대로 둘 수 있습니다. **반드시 블록 하나만 활성화(주석 해제) 상태여야 합니다.**

**3. `./run.sh`로 테스트합니다.**

### 설명은 반드시 채우세요

`설명` 필드를 비워두면 나중에 이 조합이 뭘 바꾼 건지 아무도 모릅니다. 무엇을 왜 바꿨는지
한 줄이라도 꼭 적으세요. 별도 `EXPERIMENTS.md` 문서를 안 만드는 대신 여기서 그 역할을
겸하는 거라, 설명을 안 적으면 그 기록 자체가 사라지는 셈입니다. 실패한 시도도(랩타임이
나빠졌거나 충돌/이탈이 늘었어도) 지우지 말고 주석 처리된 블록으로 남겨두세요 -- 다른
팀원이 같은 시도를 또 반복하지 않게 해줍니다.

### 작업이 끝나면 반드시 git에 올리세요

테스트가 끝났으면 성공/실패와 무관하게 바로 커밋하고 푸시하세요:

```bash
git add .
git commit -m "장애물 회피 게인 튜닝, obstacle/v2_aggressive 추가"
git push
```

## 코드 리뷰에서 발견/수정된 사항

- 코너모드 진입 디바운스가 일부 조건에만 걸려 있던 것을 전체 조건에 적용 (`lane/v1_basic.py`)
- LiDAR 후진 탈출이 후방을 체크하지 않던 것을 후방 체크 + 재시도 상한 추가 (`obstacle/v1_basic.py`)
- 신호등 `RETURN_CAMERA` 단계에서 카메라 복귀 성공 여부를 무시하던 것을 반환값 확인 + 재시도로 수정 (`traffic_light/v1_basic.py`)

## 안전 관련

- 메인 루프는 어떤 예외로 끝나든(`KeyboardInterrupt` 포함) `finally`에서 `stop_vehicle()`을
  시도합니다 -- `car_api.py` 파일 상단 설명 참고.

## 참고

- `debug_view.py`의 웹 뷰는 `assets/line-tracing/webui.html`을 찾습니다 (이 폴더 기준
  한 단계 위). 다른 위치로 배포할 때는 이 경로 가정을 확인하세요 -- 없어도 주행 자체는
  정상 동작하고 디버그 패널만 생략됩니다.
