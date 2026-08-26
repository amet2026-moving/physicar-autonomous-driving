# 최상위 FSM 심판(arbiter): 이번 프레임에 어떤 VehicleMode를 쓸지 결정.
#
# RECOVERY 모드는 없음 -- 정지/후진하느니 부딪히는 게 낫다는 팀 결정에 따라
# control/obstacle_avoid.py가 항상 회피 조향으로만 반응하기 때문(utils/states.py 참고).
# 그래서 여기 규칙은 단순하다: 장애물이 있거나(obstacle != CLEAR) 회피가 아직 진행
# 중이면(avoid_status == "ACTIVE") OBSTACLE_AVOID, 그 외엔 LANE_FOLLOW.
#
# TrafficLightState는 일부러 이 함수의 입력에 넣지 않음 -- 신호등 판단은
# decision/traffic_judge.wait_for_departure()에서 주행 루프 시작 전에 딱 한 번
# 게이트로 처리함.
from utils.states import ObstacleState, VehicleMode


def decide_mode(prev_mode: VehicleMode, lane, obstacle, avoid_status) -> VehicleMode:
    """이전 모드 + 이번 프레임 판단 결과로 다음 VehicleMode를 결정."""
    if obstacle != ObstacleState.CLEAR or avoid_status == "ACTIVE":
        return VehicleMode.OBSTACLE_AVOID
    return VehicleMode.LANE_FOLLOW
