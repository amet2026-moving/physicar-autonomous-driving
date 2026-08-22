# physicar-autonomous-driving

PhysiCar 자율주행 -- 2026 AMET 자율주행 해커톤용. 팀원별로 병렬 개발되던 여러 버전(재모
modi_test_v5/v6, Sangheon test_v7/v8, 성찬 auto3/auto4)의 검증된 로직을 하나의 협업 구조로
합친 구현입니다.

## 실행

```bash
./run.sh
```

`.venv`가 없으면 자동으로 만들고 `requirements.txt`를 설치한 뒤 `main.py`를 실행합니다.
**최초 실행 전 `config.py`의 `DRIVE_ENABLED`가 `False`인지 확인하세요** (기본값 `False`) --
`http://localhost:5000` 디버그 웹 뷰에서 BEV/차선 인식이 맞는 걸 확인한 뒤에만 `True`로
바꾸세요.

## 구조

한 서브시스템은 한 파일에 대응합니다. 다른 파일을 몰라도 자기 담당 파일만 보고 수정할 수
있도록 나눴습니다.

| 파일 | 역할 |
|---|---|
| `config.py` | 전 서브시스템 튜닝 상수 (여러 파일이 같이 쓰는 값은 `[SHARED]`로 표시) |
| `car_api.py` | PhysiCar 하드웨어 API 래퍼 -- `drive()`, `camera()`, `lidar()`, `stop_vehicle()`, `set_camera_pose()` |
| `lane_tracing.py` | 차선 인식 (순수 함수) -- BEV 변환, 흰색 경계선 슬라이딩 윈도우 추적, Stanley 조향, 노란 점선 코너 폴백, 속도 계획 |
| `lane_keeper.py` | `LaneKeeper` -- lane_tracing.py 결과로 코너모드 진입/이탈 상태머신, 조향 EMA, LOST 복구를 조합하는 컨트롤러 |
| `obstacle_avoidance.py` | `ObstacleAvoider` -- LiDAR 기반 장애물 회피 + 후진 탈출 |
| `traffic_light.py` | 신호등 인식 및 출발 처리 -- 팬/틸트 탐색(SEARCH) → 고정(LOCK) → 초록 확인 대기(WAIT_GREEN) → 주행 자세 복귀(RETURN_CAMERA) |
| `debug_view.py` | Flask 기반 디버그 웹 뷰 (튜닝/확인용, 없어도 주행에는 영향 없음) |
| `main.py` | 위 모듈을 전부 조립하는 메인 루프 -- `run.sh`가 이 파일을 실행 |

`main.py`의 흐름: 신호등 대기(`traffic_light`) → 매 프레임 카메라/라이다 동시 조회 →
`lane_keeper`로 조향/속도 계산 → `obstacle_avoidance`로 보정 → `car_api.drive()`로 명령 전송.

## 코드 리뷰에서 발견/수정된 사항

- 코너모드 진입 디바운스가 일부 조건에만 걸려 있던 것을 전체 조건에 적용 (`lane_keeper.py`)
- LiDAR 후진 탈출이 후방을 체크하지 않던 것을 후방 체크 + 재시도 상한 추가 (`obstacle_avoidance.py`)
- 신호등 `RETURN_CAMERA` 단계에서 카메라 복귀 성공 여부를 무시하던 것을 반환값 확인 + 재시도로 수정 (`traffic_light.py`)

## 안전 관련

- `DRIVE_ENABLED=False`가 기본값입니다. 디버그 뷰 확인 전에 `True`로 바꾸지 마세요.
- 메인 루프는 어떤 예외로 끝나든(`KeyboardInterrupt` 포함) `finally`에서 `stop_vehicle()`을
  시도합니다 -- `car_api.py` 파일 상단 설명 참고.

## 참고

- `debug_view.py`의 웹 뷰는 `assets/line-tracing/webui.html`을 찾습니다 (이 폴더 기준
  한 단계 위). 다른 위치로 배포할 때는 이 경로 가정을 확인하세요 -- 없어도 주행 자체는
  정상 동작하고 디버그 패널만 생략됩니다.
