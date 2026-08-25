# sensors/camera.py의 차선 기하 정보와 sensors/lidar.py의 클러스터 정보를 결합해서,
# 장애물의 좌/우를 '차량(라이다) 자체 기준'이 아니라 '차선 중심 기준'으로 판단한다.
# 차량이 차선 중앙에서 벗어나 있으면 두 기준이 서로 다른 답을 낼 수 있기 때문.
#
# T_T.py에는 이 모듈에 대응하는 코드가 없음: ConeReadOnlyDetector는 클러스터의 좌우(cy)를
# 라이다 자체 좌표계 기준으로만 판단함. 이번 설계에서 새로 추가하는 부분.
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionResult:
    """카메라+라이다 융합 결과."""
    side: str                    # "NONE"/"LEFT"/"RIGHT" -- 차선 중심 기준 장애물 위치
    distance_m: float | None = None   # 장애물까지 전방 거리 (m)
    cluster: object = None            # 판단에 쓰인 원본 라이다 클러스터


def classify_obstacle_side(lane_obs, clusters) -> FusionResult:
    """가장 가까운 유효 클러스터를 골라, 그 클러스터의 횡방향 위치를 lane_obs의
    차선 중심선(center_near/center_far) 기준으로 재투영해서 LEFT/RIGHT/NONE을 판정.
    TODO: 클러스터 원본(라이다 프레임) 좌표가 아니라 이 재투영된 오프셋으로 좌우를
    판단하도록 구현."""
    raise NotImplementedError


def draw_debug(frame, lane_obs, clusters, fusion_result: FusionResult):
    """디버그 시각화 이미지 생성 (카메라+라이다 결합 뷰) -- myapp/debug_view.py의
    패널에 합성될 예정."""
    raise NotImplementedError
