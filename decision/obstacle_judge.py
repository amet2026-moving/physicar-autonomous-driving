# sensors/fusion.py의 FusionResult를 ObstacleState로 판정하는 모듈.
#
# fusion.classify_obstacle_side()는 거리와 무관하게 콘 형상+코리도 내부 여부만 맞으면
# IN_CORRIDOR를 돌려준다. 여기서는 "실제로 반응할 만큼 가까운가"(OBSTACLE_REACT_RANGE_M)
# 를 추가로 걸러서, 멀리 있는 콘 때문에 미리 회피모드로 넘어가지 않게 한다.
from config import decision_params as cfg
from utils.states import ObstacleState


def is_obstacle_near(fusion_result, near_range_m: float) -> bool:
    """코리도 안의 장애물이 near_range_m(m) 이내로 가까이 있는지 판단."""
    return (
        fusion_result.side == "IN_CORRIDOR"
        and fusion_result.distance_m is not None
        and fusion_result.distance_m <= near_range_m
    )


def judge_obstacle_state(fusion_result) -> ObstacleState:
    """fusion_result를 ObstacleState(CLEAR/BLOCKED)로 변환."""
    if is_obstacle_near(fusion_result, cfg.OBSTACLE_REACT_RANGE_M):
        return ObstacleState.BLOCKED
    return ObstacleState.CLEAR
