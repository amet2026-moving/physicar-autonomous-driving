# 모든 모듈이 공유하는 최상위 실행 스위치와 하드웨어 제한값을 모아둔 파일.
# 특정 센서/제어기에만 쓰는 값은 여기 두지 말고 camera_params.py / lidar_params.py /
# control_params.py 로 보낼 것.

BASE_URL = "http://localhost"   # 차량 하드웨어(또는 시뮬레이터) API 서버 주소

DRIVE_ENABLED = True             # True=실제 조향/속도 명령 전송, False=인식만 하고 정지 유지 (최초 테스트는 반드시 False)

ENABLE_TRAFFIC_LIGHT_WAIT = True  # True=출발 전 신호등 GREEN 대기, False=신호등 무시하고 바로 주행 시작

TARGET_FPS = 15.0                # 목표 제어 루프 주기 (초당 프레임 수, Hz)
PRINT_INTERVAL = 1.0             # 터미널 상태 출력 주기 (초)

STEER_MAX = 20.0                 # 조향 하드웨어 물리적 한계값 (도, deg) -- 이 이상은 명령해도 무의미
STEER_TRIM_DEG = 0.0             # 조향 좌우 쏠림 보정값 (도, deg) -- 실차가 한쪽으로 틀어질 때 상수로 보정
