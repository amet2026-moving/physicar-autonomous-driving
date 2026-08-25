# 차선 원시 인식 결과(+이전 프레임의 corner_active 여부)로 LaneState를 판정하는 모듈.
#
# TODO: T_T.py main() 루프의 코너 진입/이탈 게이트를 포팅할 것
# (normal_sharp / normal_big_steer / weak_white / yellow_path_turning 각각을 N프레임
# 디바운스한 뒤에야 corner_active를 뒤집는 로직). 튜닝 상수는 config/control_params.py
# (CORNER_ENTER_*, CORNER_EXIT_*, CORNER_PATH_*, CORNER_WEAK_WHITE_*)에 있음.
from utils.states import LaneState


def is_corner(lane_obs, corner_active_prev: bool) -> bool:
    """이번 프레임이 코너 구간으로 볼만한지 판단 (디바운스 포함). TODO."""
    raise NotImplementedError


def is_off_track(lane_obs) -> bool:
    """양쪽 차선을 모두 놓쳐 경로 이탈 상태인지 판단.
    TODO: lane_obs.mode == "LOST"(양쪽 다 못찾음) 여부로 판정."""
    raise NotImplementedError


def judge_lane_state(lane_obs, corner_active_prev: bool) -> LaneState:
    """is_off_track()/is_corner() 결과를 합쳐 최종 LaneState를 결정. TODO."""
    raise NotImplementedError
