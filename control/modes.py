# 최상위 FSM 심판(arbiter): 이번 프레임에 어떤 VehicleMode를 쓸지 결정.
#
# 의도적으로 아직 미구현. utils/states.py에 VehicleMode의 세 상태(LANE_FOLLOW /
# OBSTACLE_AVOID / RECOVERY)는 확정돼 있지만, 그 사이의 전이 규칙은 아직 열린
# 설계 문제임 -- control/obstacle_avoid.py와 control/recovery.py가 만들어져서
# "제어권을 넘겨야 하는지/에스컬레이션해야 하는지"를 알려줄 수 있게 된 뒤에 채울 것.
#
# 예상되는 형태 스케치 (최종안 아님, 참고용):
#
#     def decide_mode(prev_mode, lane, obstacle, avoid_status):
#         if avoid_status == "NEEDS_RECOVERY":
#             return VehicleMode.RECOVERY
#         if obstacle != ObstacleState.CLEAR:
#             return VehicleMode.OBSTACLE_AVOID
#         return VehicleMode.LANE_FOLLOW
#
# 참고: TrafficLightState는 일부러 이 함수의 입력에 넣지 않음 -- 신호등 판단은
# decision/traffic_judge.wait_for_green()에서 주행 루프 시작 전에 딱 한 번 게이트로
# 처리함 (light_1.py의 wait_for_start_gate()와 동일한 방식).
from utils.states import VehicleMode


def decide_mode(prev_mode: VehicleMode, lane, obstacle, avoid_status) -> VehicleMode:
    """이전 모드 + 이번 프레임 판단 결과로 다음 VehicleMode를 결정. TODO (모듈 상단 참고)."""
    raise NotImplementedError
