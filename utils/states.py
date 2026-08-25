# 모든 FSM(상태머신) 상태 정의를 한곳에 모아둔 파일. 상태를 "계산/판단"하는 코드와
# 상태 "정의"를 분리해서, 상태 이름/종류가 바뀔 때 이 파일 하나만 보면 되게 한다.
#
# - TrafficLightState / LaneState / ObstacleState : decision/ 아래 판단 함수들이 만들어냄.
# - VehicleMode : 최상위 제어 상태. 전이 규칙은 아직 미구현 -- control/modes.py 참고.
#   지금은 상태 이름(종류)만 확정되어 있음.
from enum import Enum


class TrafficLightState(Enum):
    """신호등 인식 결과 상태."""
    RED = "RED"          # 빨간불 -- 정지 유지
    GREEN = "GREEN"       # 초록불 -- 출발 가능
    UNKNOWN = "UNKNOWN"   # 인식 실패 -- RED와 동일하게(정지) 취급


class LaneState(Enum):
    """차선 인식 결과로 판단한 주행 구간 상태."""
    STRAIGHT = "STRAIGHT"     # 직선 구간
    CORNER = "CORNER"          # 코너(커브) 구간
    OFF_TRACK = "OFF_TRACK"    # 양쪽 차선 모두 인식 실패(경로 이탈)


class ObstacleState(Enum):
    """카메라+라이다 융합 결과로 판단한 장애물 상태 (차선 중심 기준 좌/우)."""
    CLEAR = "CLEAR"    # 장애물 미인식
    LEFT = "LEFT"        # 장애물이 차선 중심 기준 좌측에 있음
    RIGHT = "RIGHT"       # 장애물이 차선 중심 기준 우측에 있음


class VehicleMode(Enum):
    """최상위 제어 모드. 전이 규칙은 TODO -- control/modes.py 참고."""
    LANE_FOLLOW = "LANE_FOLLOW"          # 기본 주행 (직선/코너 모두 포함, 내부에서 알고리즘 분기)
    OBSTACLE_AVOID = "OBSTACLE_AVOID"     # 장애물 회피 중
    RECOVERY = "RECOVERY"                  # 막힘 -> 후진 후 재탐색 중
