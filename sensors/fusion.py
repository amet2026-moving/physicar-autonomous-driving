# sensors/camera.py의 차선 기하 정보와 sensors/lidar.py의 클러스터 정보를 결합해서,
# 장애물의 좌/우를 '차량(라이다) 자체 기준'이 아니라 '차선 기준'으로 판단한다.
# 차량이 차선 중앙에서 벗어나 있으면 두 기준이 서로 다른 답을 낼 수 있기 때문.
#
# 차선 기준선은 노란 점선을 우선 사용하고(camera.py는 코너 여부와 무관하게 항상
# 노란선을 인식해둠), 노란선이 안 보이면 흰선 좌우 중심선(center_near/center_far)으로
# 대체한다.
#
# T_T.py에는 이 모듈에 대응하는 코드가 없음: ConeReadOnlyDetector는 클러스터의 좌우(cy)를
# 라이다 자체 좌표계 기준으로만 판단했다. 실험적으로는 AutoV3_fusion.py(임시 몽키패치
# 버전)에서 같은 아이디어를 검증했고, 이 파일은 그 판정 수식을 이 프로젝트 구조에 맞게
# 정식으로 이식한 것이다.
#
# 카메라<->라이다 외부 캘리브레이션(정확한 물리적 장착 위치/각도 차이)이 없어서, 절대
# 좌표 변환 대신 "각자 자기 기준에서 중심 대비 얼마나 벗어났는지 비율"로 정규화해서
# 비교한다 (AutoV3_fusion.py와 동일한 방식). 튜닝 상수는 config/fusion_params.py에 있음.
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import fusion_params as fcfg
from config import lidar_params as lidar_cfg
from sensors import lidar


@dataclass
class FusionResult:
    """카메라+라이다 융합 결과."""
    side: str                    # "NONE"/"LEFT"/"RIGHT" -- 차선 기준 장애물 위치
    distance_m: float | None = None   # 장애물까지 전방 거리 (m)
    cluster: object = None            # 판단에 쓰인 원본 라이다 클러스터


# draw_debug()가 classify_obstacle_side()의 중간 계산값(기준선 종류/비율 등)을
# 다시 계산하지 않고 쓰도록 저장해두는 캐시. classify_obstacle_side() 호출 시마다
# 새로 갱신됨 (camera.py의 _last_debug와 같은 패턴).
_last_debug = {}


# ============================================================
# LANE_REFERENCE -- 차선 기준선 고르기 (노란 점선 우선, 없으면 흰선 중심선)
# ============================================================

def _lane_reference_points(lane_obs):
    """차선 기준선을 이루는 점 목록을 반환: [(x,y), ...] (BEV px, y는 아무 순서나 가능).
    노란 점선(yellow_path)이 있으면 그걸 우선 쓰고, 없으면 흰선 좌우 중심선
    (center_near/center_far)으로 대체한다. 둘 다 없으면 (None, "NONE")."""
    if not lane_obs.bev_w:
        return None, "NONE"

    if lane_obs.yellow_path:
        # 노란점이 1개뿐이어도 기준선이 되도록, 차량 위치(화면 하단 중앙)를 첫
        # 점으로 붙여서 최소 2점짜리 선을 만든다. build_yellow_path()가 경로를
        # 시작할 때 쓰는 차량 위치 근사(화면 하단 중앙)와 같은 발상.
        vehicle_anchor = (lane_obs.bev_w / 2.0, lane_obs.y_near)
        return [vehicle_anchor] + list(lane_obs.yellow_path), "YELLOW"

    if lane_obs.center_near is not None and lane_obs.center_far is not None:
        return [(lane_obs.center_near, lane_obs.y_near), (lane_obs.center_far, lane_obs.y_far)], "WHITE_CENTER"

    return None, "NONE"


def _interp_x_at_row(points, target_y):
    """기준선 점들을 BEV y좌표 순으로 정렬한 뒤, target_y에서의 x를 선형보간.
    target_y가 점들의 범위를 벗어나면 가장 가까운 끝점 값으로 고정(clamp)된다."""
    pts = sorted(points, key=lambda p: -p[1])   # y 내림차순(가까운 점 먼저)
    neg_ys = [-p[1] for p in pts]                # np.interp는 x가 증가해야 하므로 부호 반전
    xs = [p[0] for p in pts]
    return float(np.interp(-target_y, neg_ys, xs))


def _target_row_for_distance(cx_m, lane_obs):
    """장애물의 전방거리(m)를, 그 거리에 해당하는 BEV 행(y좌표)으로 근사 변환.
    카메라<->라이다 캘리브레이션이 없어 LIDAR_CLUSTER_MAX_RANGE_M을 y_far에,
    거리 0을 y_near에 대응시키는 선형 근사를 쓴다."""
    dist_ratio = float(np.clip(cx_m / lidar_cfg.LIDAR_CLUSTER_MAX_RANGE_M, 0.0, 1.0))
    return lane_obs.y_near + (lane_obs.y_far - lane_obs.y_near) * dist_ratio


# ============================================================
# OBSTACLE_SIDE_CLASSIFICATION -- 장애물 좌/우를 차선 기준으로 판정
# ============================================================

def classify_obstacle_side(lane_obs, clusters) -> FusionResult:
    """가장 가까운 유효 클러스터(콘 형상 필터 통과)를 골라, 차선 기준선
    (노란 점선 우선, 없으면 흰선 중심선) 대비 좌/우를 판정한다.

    판정 방법: 카메라 쪽(차선 기준선이 화면 중심에서 벗어난 비율)과 라이다 쪽
    (장애물이 차량 정면축에서 벗어난 비율)을 각각 계산해서 뺀다 -- 둘 다
    "중심 대비 비율"이라는 같은 단위로 맞춘 뒤 비교하는 것이므로, 차량이 차선
    중앙에서 벗어나 있어도(두 기준이 다른 값을 가리켜도) 올바르게 상쇄된다."""
    _last_debug.clear()

    valid = lidar.classify_cone_candidates(clusters, corner_hint=False)
    if not valid:
        return FusionResult(side="NONE")

    cluster = min(valid, key=lambda c: c.dmin)
    ref_points, source = _lane_reference_points(lane_obs)
    _last_debug["cluster"] = cluster
    _last_debug["reference_points"] = ref_points
    _last_debug["reference_source"] = source

    if ref_points is None:
        _last_debug["reason"] = "NO_REFERENCE"
        return FusionResult(side="NONE", distance_m=cluster.cx, cluster=cluster)

    if abs(cluster.cy) < fcfg.FUSION_CY_DEADBAND_M:
        _last_debug["reason"] = "CENTERED"
        return FusionResult(side="NONE", distance_m=cluster.cx, cluster=cluster)

    half_w = lane_obs.bev_w / 2.0
    target_row = _target_row_for_distance(cluster.cx, lane_obs)
    lane_x = _interp_x_at_row(ref_points, target_row)

    # 카메라 쪽: 차선 기준선이 화면 중심에서 얼마나 벗어났는지 (+ = 오른쪽).
    lane_offset_ratio = (lane_x - half_w) / half_w
    # 라이다 쪽: 장애물이 차량 정면축에서 얼마나 벗어났는지, 같은 비율 단위로.
    # cy는 +좌/-우이므로 부호를 뒤집어 이미지 좌표계(+우)로 맞춘다.
    cone_offset_ratio = float(np.clip(-cluster.cy / fcfg.FUSION_LATERAL_HALF_WIDTH_M, -1.5, 1.5))
    relative = cone_offset_ratio - lane_offset_ratio

    if relative > fcfg.FUSION_SIDE_DEADBAND_RATIO:
        side = "RIGHT"
    elif relative < -fcfg.FUSION_SIDE_DEADBAND_RATIO:
        side = "LEFT"
    else:
        side = "NONE"

    _last_debug.update({
        "lane_x": lane_x, "target_row": target_row,
        "lane_offset_ratio": lane_offset_ratio,
        "cone_offset_ratio": cone_offset_ratio,
        "relative": relative,
    })

    return FusionResult(side=side, distance_m=cluster.cx, cluster=cluster)


# ============================================================
# DEBUG VISUALIZATION -- BEV 오버레이 + 라이다 레이더뷰 + 텍스트 3분할
# ============================================================

def _reproject_cluster_px(cluster, lane_obs):
    """클러스터의 (cx,cy)를 BEV 픽셀 좌표로 근사 변환 (차선 기준과 무관하게,
    클러스터 자신의 차량-정면축 대비 위치만으로). draw_debug()의 점 찍기용."""
    if not lane_obs.bev_w:
        return None
    half_w = lane_obs.bev_w / 2.0
    ratio = float(np.clip(-cluster.cy / fcfg.FUSION_LATERAL_HALF_WIDTH_M, -1.5, 1.5))
    px = int(np.clip(half_w + ratio * half_w, 0, lane_obs.bev_w - 1))
    py = int(_target_row_for_distance(cluster.cx, lane_obs))
    return px, py


def draw_debug(frame, lane_obs, clusters, fusion_result: FusionResult) -> np.ndarray:
    """카메라(차선 기준선)+라이다(클러스터) 결합 결과를 3분할로 시각화.
    frame은 원본이 아니라 BEV 프레임이어야 한다 -- 판정 자체가 BEV 픽셀 공간에서
    일어나므로. classify_obstacle_side()를 먼저 호출한 뒤 불러야 _last_debug
    캐시(기준선/비율 등)가 채워져 있다."""
    h, w = frame.shape[:2]

    # ---- 패널 1: BEV + 차선 기준선 + 클러스터 오버레이 ----
    bev_vis = frame.copy()

    ref_points = _last_debug.get("reference_points")
    source = _last_debug.get("reference_source", "NONE")
    if ref_points:
        line_color = (255, 0, 255) if source == "YELLOW" else (0, 255, 255)   # magenta / cyan
        pts = np.array([(int(x), int(y)) for x, y in sorted(ref_points, key=lambda p: -p[1])], dtype=np.int32)
        cv2.polylines(bev_vis, [pts], False, line_color, 2, cv2.LINE_AA)

    valid_ids = {id(c) for c in lidar.classify_cone_candidates(clusters, corner_hint=False)}
    chosen = fusion_result.cluster

    for c in clusters:
        pos = _reproject_cluster_px(c, lane_obs)
        if pos is None:
            continue
        px, py = pos
        if chosen is not None and c is chosen:
            color, radius = (0, 0, 255), 9        # red -- 이번에 선택된 장애물
        elif id(c) in valid_ids:
            color, radius = (0, 165, 255), 6       # orange -- 콘 후보지만 선택 안 됨
        else:
            color, radius = (150, 150, 150), 4     # gray -- 형상 필터 통과 못함
        cv2.circle(bev_vis, (px, py), radius, color, -1)

    if chosen is not None:
        px, py = _reproject_cluster_px(chosen, lane_obs)
        cv2.circle(bev_vis, (px, py), 12, (255, 255, 255), 2)
        cv2.putText(bev_vis, f"{fusion_result.side} {fusion_result.distance_m:.2f}m",
                    (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    if lane_obs.bev_w:
        cv2.circle(bev_vis, (int(lane_obs.bev_w / 2), lane_obs.y_near), 6, (255, 255, 255), -1)

    cv2.putText(bev_vis, f"FUSION  ref={source}  magenta=yellow cyan=white-center",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- 패널 2: 라이다 위에서 본 뷰 (sensors/lidar.py의 draw_debug 재사용) ----
    radar = cv2.resize(lidar.draw_debug(clusters), (w, h), interpolation=cv2.INTER_AREA)

    # ---- 패널 3: 텍스트 정보 ----
    info = np.zeros((h, w, 3), dtype=np.uint8)
    lines = [f"reference source: {source}"]

    if chosen is None:
        lines.append("chosen cluster: none (no cone-shaped cluster)")
    else:
        lines.append(f"cx={chosen.cx:+.2f}m  cy={chosen.cy:+.2f}m  dmin={chosen.dmin:.2f}m")
        reason = _last_debug.get("reason")
        if reason:
            lines.append(f"side=NONE (reason={reason})")
        elif "relative" in _last_debug:
            lines += [
                f"lane_offset_ratio={_last_debug['lane_offset_ratio']:+.3f}",
                f"cone_offset_ratio={_last_debug['cone_offset_ratio']:+.3f}",
                f"relative={_last_debug['relative']:+.3f}  deadband={fcfg.FUSION_SIDE_DEADBAND_RATIO}",
                f"side={fusion_result.side}",
            ]

    for i, text in enumerate(lines):
        cv2.putText(info, text, (18, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return np.hstack([bev_vis, radar, info])


# ============================================================
# 단독 실행: 카메라+라이다 융합 결과만 웹 디버그 뷰로 확인
# ============================================================
if __name__ == "__main__":
    import time

    from config import base as base_cfg
    from hardware import car_api
    from myapp import debug_view
    from sensors import camera

    debug_view.serve()
    print("[fusion.py] perception-only self test -- http://localhost:5000")

    try:
        while True:
            t0 = time.time()
            frame = car_api.camera()
            if frame is None:
                time.sleep(0.2)
                continue

            bev, _M = camera.warp_to_bev(frame)
            lane_obs = camera.detect_lane_lines(bev)

            scan = lidar.capture_scan()
            clusters = lidar.build_clusters(scan)

            fusion_result = classify_obstacle_side(lane_obs, clusters)
            panel = draw_debug(bev, lane_obs, clusters, fusion_result)

            status = (
                f"ref={_last_debug.get('reference_source', 'NONE')} "
                f"side={fusion_result.side} dist={fusion_result.distance_m}"
            )
            debug_view.update_web(panel, status)

            proc_time = time.time() - t0
            time.sleep(max(0.0, 1.0 / base_cfg.TARGET_FPS - proc_time))
    except KeyboardInterrupt:
        print("\n[fusion.py] stopped")
    finally:
        lidar.stop_scan()
        debug_view.stop_view()
