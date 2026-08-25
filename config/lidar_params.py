# 라이다 수신, 클러스터링, 콘(라바콘) 형상 인식 관련 값 모음.
#
# 여기는 '센싱'만 다룸 -- "무엇을 하나의 덩어리(클러스터)로 볼지", "어떤 모양이 콘처럼
# 보이는지"만 정의. 확정된 장애물에 어떻게 반응할지(회피 오프셋, 속도, 후진 거리 등)는
# control_params.py에 있음.

LIDAR_TOPIC = "/scan"   # ROS2 라이다 스캔 토픽 이름

# ============================================================
# 클러스터링 (인접한 포인트를 하나의 물체로 묶기)
# ============================================================
LIDAR_CLUSTER_HZ = 10.0                # 클러스터링 처리 주기 (Hz)
LIDAR_CLUSTER_FOV_DEG = 100.0          # 클러스터링 대상 시야각 (도, 전방 기준 좌우 각각)
LIDAR_CLUSTER_MAX_RANGE_M = 2.0        # 클러스터링 대상 최대 거리 (m)
LIDAR_CLUSTER_LINK_M = 0.12            # 인접 포인트를 같은 클러스터로 묶을 최대 간격 (m)
LIDAR_CLUSTER_MIN_POINTS = 3           # 유효 클러스터로 인정할 최소 포인트 수 (개)
LIDAR_CLUSTER_MAX_PRINT = 8            # 터미널에 출력할 클러스터 최대 개수 (개)

# ============================================================
# 콘 형상 판정 (직선 구간 탐색)
# ============================================================
CONE_WIDTH_MIN_M = 0.07                # 콘으로 볼 최소 폭 (m)
CONE_WIDTH_MAX_M = 0.20                # 콘으로 볼 최대 폭 (m)
CONE_MIN_POINTS = 5                    # 콘 후보 최소 포인트 수 (개)
CONE_MAX_ANGLE_SPAN_DEG = 30.0         # 콘 후보 최대 각도 폭 (도)

CONE_DETECT_MAX_RANGE_M = 1.50         # 새 콘을 탐지할 최대 거리 (m)
CONE_SEARCH_ANGLE_DEG = 35.0           # 콘 탐색 각도 범위 (도, 전방 기준 좌우)
CONE_SEARCH_MAX_ABS_Y_M = 0.55         # 콘 탐색 최대 횡방향 거리 (m)
CONE_CONFIRM_FRAMES = 2                # 콘으로 확정하는 데 필요한 연속 프레임 수 (개)

CONE_TRACK_MAX_RANGE_M = 1.80          # 추적 중인 콘을 계속 추적할 최대 거리 (m)
CONE_TRACK_ANGLE_DEG = 100.0           # 추적 허용 각도 범위 (도)
CONE_TRACK_MAX_CENTER_SHIFT_M = 0.45   # 프레임간 콘 중심 최대 이동 허용치 (m) -- 초과하면 다른 물체로 간주

CONE_PASSING_ANGLE_DEG = 45.0          # 콘을 '지나쳤다'고 볼 각도 기준 (도)
CONE_LOST_FRAMES = 3                   # 콘을 놓쳤다고 판정할 연속 미탐지 프레임 수 (개)

# ============================================================
# 콘 형상 판정 (코너 구간 탐색 -- 직선보다 넓은 게이트)
# ============================================================
CONE_CORNER_DETECT_MAX_RANGE_M = 1.95  # 코너 구간 콘 탐지 최대 거리 (m)
CONE_CORNER_SEARCH_ANGLE_DEG = 75.0    # 코너 구간 콘 탐색 각도 범위 (도)
CONE_CORNER_SEARCH_MAX_ABS_Y_M = 0.85  # 코너 구간 콘 탐색 최대 횡방향 거리 (m)

CONE_CORNER_WIDTH_MIN_M = 0.04         # 코너 구간 콘 최소 폭 (m)
CONE_CORNER_WIDTH_MAX_M = 0.26         # 코너 구간 콘 최대 폭 (m)
CONE_CORNER_MAX_ANGLE_SPAN_DEG = 40.0  # 코너 구간 콘 후보 최대 각도 폭 (도)
CONE_CORNER_MIN_POINTS = 4             # 코너 구간 콘 후보 최소 포인트 수 (개)

CONE_CORNER_URGENT_RANGE_M = 0.0                # 코너 구간 긴급 회피 판단 거리 (m, 0.0=미사용)
CONE_CORNER_PASS_X_M = 0.08                     # 코너 구간 콘 통과 판정 전방거리 (m)
CONE_CORNER_TRACK_MAX_CENTER_SHIFT_M = 0.65     # 코너 구간 콘 추적 중심이동 허용치 (m)

# ============================================================
# 원시 충돌 형상 인식 (콘이 아니어도 부딪힐 수 있는 물체 전반)
# ============================================================
RAW_SHIELD_MIN_WIDTH_M = 0.04          # 충돌 위험 형상 최소 폭 (m)
RAW_SHIELD_MAX_WIDTH_M = 0.24          # 충돌 위험 형상 최대 폭 (m)
RAW_SHIELD_MAX_ANGLE_SPAN_DEG = 25.0   # 충돌 위험 형상 최대 각도 폭 (도)
RAW_SHIELD_MIN_POINTS = 4              # 충돌 위험 형상 최소 포인트 수 (개)
