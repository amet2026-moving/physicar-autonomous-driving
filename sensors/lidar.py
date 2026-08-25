# 라이다 수신 + 클러스터링(포인트 뭉치 찾기) + 콘 형상 분류.
#
# TODO: T_T.py에서 아래를 포팅할 것
#   - _build_lidar_clusters() -> build_clusters()
#   - _cone_search_candidate() / _cone_track_candidate() -> classify_cone_candidates()
#   - ConeReadOnlyDetector의 SEARCH -> CANDIDATE -> DETECTED -> PASSING -> NONE 추적 로직
#     (프레임간 추적이 sensors/lidar.py가 아니라 control/obstacle_avoid.py 쪽 책임이
#     될 수도 있음 -- 채우면서 결정할 것)
#
# 채워넣을 때 결정할 것: T_T.py는 rclpy로 /scan 토픽을 직접 구독(백그라운드 ROS2 노드 +
# 스레드)하지만, 이전 car_api.py는 단순 HTTP GET /lidar 폴링 방식이었음
# (hardware.car_api.lidar()로 이미 사용 가능). 레이턴시/주기 문제로 꼭 필요한 게
# 아니라면 더 단순한 HTTP 방식부터 시작할 것.
#
# 튜닝 상수는 config/lidar_params.py에 있음.
from dataclasses import dataclass


@dataclass
class Cluster:
    """라이다 포인트들을 하나로 묶은 클러스터(물체 후보) 하나."""
    n: int          # 클러스터에 속한 포인트 개수 (개)
    dmin: float     # 클러스터 내 최소 거리 (m)
    dmed: float     # 클러스터 내 중간값 거리 (m)
    angle: float    # 클러스터 중심 각도 (도, +=좌측)
    x_min: float    # 전방(x) 최소값 (m)
    x_max: float    # 전방(x) 최대값 (m)
    y_min: float    # 횡방향(y) 최소값 (m)
    y_max: float    # 횡방향(y) 최대값 (m)
    width: float    # 클러스터 폭 (대각선 길이, m)


def capture_scan():
    """원시 라이다 스캔 수신. TODO: hardware.car_api.lidar() 또는 rclpy /scan
    구독 중 선택 (모듈 상단 설명 참고)."""
    raise NotImplementedError


def build_clusters(scan) -> list[Cluster]:
    """원시 스캔을 인접 포인트끼리 묶어 클러스터 리스트로 변환.
    TODO: T_T.py의 _build_lidar_clusters() 포팅."""
    raise NotImplementedError


def classify_cone_candidates(clusters: list[Cluster], corner_hint: bool = False) -> list[Cluster]:
    """클러스터 중 콘(라바콘) 형상에 맞는 것만 골라냄.
    corner_hint: 코너 구간이면 더 넓은 탐색 게이트를 쓰기 위한 힌트.
    TODO: T_T.py의 _cone_search_candidate()/_cone_track_candidate() 포팅."""
    raise NotImplementedError


def draw_debug(clusters: list[Cluster]):
    """디버그 시각화 이미지 생성 -- myapp/debug_view.py의 패널에 합성될 예정."""
    raise NotImplementedError
