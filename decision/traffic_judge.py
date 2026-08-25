# 신호등 원시 인식 결과를 TrafficLightState로 판정하고, 주행 루프 시작 전 신호대기
# 게이트를 담당하는 모듈.
#
# TODO: light_1.py의 StartGateController를 포팅할 것 (RED/GREEN 점수 계산 +
# GREEN_CONFIRM_FRAMES/RED_CLEAR_DEPART_SEC 디바운스, 그리고 wait_for_start_gate()의
# 위치락 탐색 루프). 튜닝 상수는 config/camera_params.py에 있음
# (TRAFFIC_SCORE_MIN, GREEN_CONFIRM_FRAMES 등).
from utils.states import TrafficLightState


def is_green(traffic_obs) -> bool:
    """이번 프레임 관측이 GREEN으로 확정할만한지 판단. TODO."""
    raise NotImplementedError


def judge_traffic_light(traffic_obs) -> TrafficLightState:
    """is_green() 판단에 RED/UNKNOWN 처리까지 더해 최종 상태 결정. TODO."""
    raise NotImplementedError


def wait_for_green() -> None:
    """주행 루프 시작 전, main.py에서 딱 한 번만 호출되는 블로킹 대기 함수
    (light_1.py의 wait_for_start_gate()와 동일한 역할). 매 프레임 도는 VehicleMode
    FSM에는 속하지 않음 -- 프로그램 실행당 한 번만 지나가는 단계.

    TODO: config.camera_params의 DRIVE_CAMERA_PAN/TILT_DEG로 카메라를 고정하고,
    신호등 위치를 락(lock)한 뒤, is_green()이 True가 될 때까지 반복.
    (위치를 끝내 못 찾으면 NO_LOCK_TIMEOUT_SEC 후 강제 출발 -- light_1.py의
    상태없는 폴백과 동일)."""
    raise NotImplementedError
