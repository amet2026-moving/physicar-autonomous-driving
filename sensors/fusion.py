# sensors/camera.py의 차선 기하 정보와 sensors/lidar.py의 클러스터 정보를 결합해서,
# 장애물이 '지금 내가 달리는 코리도(왼쪽 흰선~노란선)' 안에 들어와 있는지 판단한다.
# 회피 방향은 항상 오른쪽으로 고정(팀 결정)이라 좌/우를 가릴 필요가 없다 -- 노란선
# 건너 반대 차로를 지나가는 물체는 코리도 밖이므로 무시해야, 반대 차로 통행마다
# 헛되이 회피 모드로 들어가지 않는다.
#
# 코리도 기준선은 camera.py의 LaneTracker가 이미 만들어둔 lane_obs.center_near/far
# (왼쪽흰선+노란선 중점)와 lane_width_px(코리도 폭 EMA)를 그대로 재사용한다 -- 그게
# 곧 지금 차가 따라가고 있는 주행선이므로, 여기서 별도의 기준선을 다시 고를 필요가
# 없다.
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
    side: str                    # "NONE"/"IN_CORRIDOR" -- 지금 주행 코리도 내부 여부
    distance_m: float | None = None   # 장애물까지 전방 거리 (m)
    cluster: object = None            # 판단에 쓰인 원본 라이다 클러스터


# draw_debug()가 classify_obstacle_side()의 중간 계산값(기준선 종류/비율 등)을
# 다시 계산하지 않고 쓰도록 저장해두는 캐시. classify_obstacle_side() 호출 시마다
# 새로 갱신됨 (camera.py의 _last_debug와 같은 패턴).
_last_debug = {}


# ============================================================
# 기하 변환 헬퍼 -- 장애물 전방거리 <-> BEV 행(row), 코리도 중심선 보간
# ============================================================

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
# OBSTACLE_CORRIDOR_CLASSIFICATION -- 장애물이 주행 코리도 안에 있는지 판정
# ============================================================

def classify_obstacle_side(lane_obs, clusters) -> FusionResult:
    """가장 가까운 유효 클러스터(콘 형상 필터 통과)를 골라, 지금 주행 코리도
    (lane_obs.center_near/far ± lane_width_px/2, 즉 왼쪽흰선~노란선) 안에 들어와
    있는지 판정한다. 회피 방향은 항상 오른쪽으로 고정(팀 결정)이라 좌/우를 가릴
    필요가 없다 -- 노란선 건너 반대 차로를 지나가는 물체는 코리도 밖이라 무시된다."""
    _last_debug.clear()

    valid = lidar.classify_cone_candidates(clusters, corner_hint=False)
    if not valid:
        return FusionResult(side="NONE")

    cluster = min(valid, key=lambda c: c.dmin)
    _last_debug["cluster"] = cluster

    if abs(cluster.cy) < fcfg.FUSION_CY_DEADBAND_M:
        # 라이다 정면축 거의 정중앙 -- 코리도 기준선을 볼 것도 없이 명백히 막고 있음.
        _last_debug["reason"] = "CENTERED"
        return FusionResult(side="IN_CORRIDOR", distance_m=cluster.cx, cluster=cluster)

    if lane_obs.center_near is None or lane_obs.center_far is None or not lane_obs.bev_w:
        _last_debug["reason"] = "NO_REFERENCE"
        return FusionResult(side="NONE", distance_m=cluster.cx, cluster=cluster)

    target_row = _target_row_for_distance(cluster.cx, lane_obs)
    lane_x = _interp_x_at_row(
        [(lane_obs.center_near, lane_obs.y_near), (lane_obs.center_far, lane_obs.y_far)],
        target_row,
    )
    corridor_half_px = 0.5 * (
        lane_obs.lane_width_px
        if lane_obs.lane_width_px is not None
        else fcfg.FUSION_DEFAULT_CORRIDOR_WIDTH_RATIO * lane_obs.bev_w
    )
    margin_px = fcfg.FUSION_CORRIDOR_MARGIN_RATIO * corridor_half_px

    obstacle_px, _ = _reproject_cluster_px(cluster, lane_obs)
    in_corridor = (lane_x - corridor_half_px - margin_px) <= obstacle_px <= (lane_x + corridor_half_px + margin_px)
    side = "IN_CORRIDOR" if in_corridor else "NONE"

    _last_debug.update({
        "lane_x": lane_x, "target_row": target_row,
        "corridor_half_px": corridor_half_px, "obstacle_px": obstacle_px,
        "in_corridor": in_corridor,
    })

    return FusionResult(side=side, distance_m=cluster.cx, cluster=cluster)


# ============================================================
# DEBUG VISUALIZATION -- BEV 오버레이 + 라이다 레이더뷰 + 텍스트 3분할
# ============================================================

def _reproject_cluster_px(cluster, lane_obs):
    """클러스터의 (cx,cy)를 BEV 픽셀 좌표로 근사 변환 (차선 기준과 무관하게,
    클러스터 자신의 차량-정면축 대비 위치만으로). draw_debug()의 점 찍기와
    classify_obstacle_side()의 코리도 내부 판정에 공용으로 쓴다."""
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

    # ---- 패널 1: BEV + 주행 코리도(왼쪽흰선~노란선) + 클러스터 오버레이 ----
    bev_vis = frame.copy()

    if lane_obs.center_near is not None and lane_obs.center_far is not None:
        center_pts = np.array(
            [(int(lane_obs.center_near), lane_obs.y_near), (int(lane_obs.center_far), lane_obs.y_far)],
            dtype=np.int32,
        )
        cv2.polylines(bev_vis, [center_pts], False, (0, 255, 255), 2, cv2.LINE_AA)   # cyan -- 코리도 중심선

        half_px = _last_debug.get("corridor_half_px")
        if half_px is not None:
            for sign, color in ((-1, (0, 255, 0)), (+1, (255, 0, 255))):   # green=왼쪽경계 magenta=오른쪽경계(노란선)
                bound_pts = np.array([
                    (int(lane_obs.center_near + sign * half_px), lane_obs.y_near),
                    (int(lane_obs.center_far + sign * half_px), lane_obs.y_far),
                ], dtype=np.int32)
                cv2.polylines(bev_vis, [bound_pts], False, color, 1, cv2.LINE_AA)

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

    cv2.putText(bev_vis, "FUSION  cyan=corridor center  green=left bound  magenta=yellow bound",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- 패널 2: 라이다 위에서 본 뷰 (sensors/lidar.py의 draw_debug 재사용) ----
    radar = cv2.resize(lidar.draw_debug(clusters), (w, h), interpolation=cv2.INTER_AREA)

    # ---- 패널 3: 텍스트 정보 ----
    info = np.zeros((h, w, 3), dtype=np.uint8)
    lines = [f"lane mode: {lane_obs.mode}"]

    if chosen is None:
        lines.append("chosen cluster: none (no cone-shaped cluster)")
    else:
        lines.append(f"cx={chosen.cx:+.2f}m  cy={chosen.cy:+.2f}m  dmin={chosen.dmin:.2f}m")
        reason = _last_debug.get("reason")
        if reason:
            lines.append(f"side={fusion_result.side} (reason={reason})")
        elif "in_corridor" in _last_debug:
            lines += [
                f"lane_x={_last_debug['lane_x']:.1f}  obstacle_px={_last_debug['obstacle_px']:.1f}",
                f"corridor_half_px={_last_debug['corridor_half_px']:.1f}",
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

            status = f"lane_mode={lane_obs.mode} side={fusion_result.side} dist={fusion_result.distance_m}"
            debug_view.update_web(panel, status)

            proc_time = time.time() - t0
            time.sleep(max(0.0, 1.0 / base_cfg.TARGET_FPS - proc_time))
    except KeyboardInterrupt:
        print("\n[fusion.py] stopped")
    finally:
        lidar.stop_scan()
        debug_view.stop_view()
