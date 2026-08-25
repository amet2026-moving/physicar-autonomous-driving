# sensors/fusion.py의 FusionResult를 ObstacleState로 판정하는 모듈.
#
# TODO: fusion_result.distance_m을 근접거리 임계값(near_range_m, 필요하면
# config/control_params.py에 상수 추가)과 비교하고, fusion_result.side
# ("LEFT"/"RIGHT"/"NONE")를 ObstacleState로 매핑할 것.
from utils.states import ObstacleState


def is_obstacle_near(fusion_result, near_range_m: float) -> bool:
    """장애물이 near_range_m(m) 이내로 가까이 있는지 판단. TODO."""
    raise NotImplementedError


def judge_obstacle_state(fusion_result) -> ObstacleState:
    """fusion_result를 ObstacleState(CLEAR/LEFT/RIGHT)로 변환. TODO."""
    raise NotImplementedError
