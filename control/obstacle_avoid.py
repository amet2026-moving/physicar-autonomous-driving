# VehicleMode.OBSTACLE_AVOID용 회피 컨트롤러.
#
# TODO: T_T.py의 ConeAvoidanceFSM을 포팅할 것. 내부 5단계
# (NORMAL -> SHIFT_OUT -> PASS -> SHIFT_IN -> RECOVER -> NORMAL)는 이 클래스
# 내부에만 있는 구현 디테일로 숨길 것 -- control/modes.py에는 세부 5단계가 아니라
# step()이 반환하는 대략적인 status 문자열만 보여줘야 함. 그래야 최상위 FSM은
# 단순하게 유지하면서, 검증된 5단계 기하 로직은 그대로 재사용할 수 있음.
# 튜닝 상수는 config/control_params.py (AVOID_* / ROAD_GUARD_*)에 있음.


class ObstacleAvoidController:
    """장애물 회피 상태를 프레임간에 들고 있는 컨트롤러 (인스턴스 하나를 매 프레임 재사용)."""

    def __init__(self):
        raise NotImplementedError

    def step(self, obstacle_state, lane_obs) -> tuple[float, float, str]:
        """이번 프레임의 (조향각(도), 속도(m/s), status)를 반환.
        status는 "ACTIVE"(회피 진행중) / "DONE"(회피 완료) / "NEEDS_RECOVERY"(후진 필요) 중 하나.
        TODO."""
        raise NotImplementedError
