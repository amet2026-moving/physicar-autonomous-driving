# 라이다 수신(rclpy /scan 구독) + 클러스터링(포인트 뭉치 찾기) + 콘 형상 분류.
#
# 이 파일도 camera.py와 같은 원칙: "무엇이 보이는가"만 답한다. 클러스터가 콘 모양인지
# 형태만 판정하고(classify_cone_candidates), 그 콘을 여러 프레임에 걸쳐 추적/래칭하는
# 것은 여기 없음 -- sensors/fusion.py가 매 프레임 독립적으로 "가장 가까운 유효
# 클러스터"를 고르고, 그걸 프레임 간에 어떻게 다룰지(같은 물체로 볼지, 회피 방향을
# 얼마나 유지할지)는 control/obstacle_avoid.py, control/recovery.py가 각자 컨트롤러
# 상태로 들고 있는다.
#
# 원본: T_T.py의 LiDAR 클러스터 로거 + 콘 형상 판정 로직을 그대로 포팅. 튜닝 상수는
# config/lidar_params.py에 있음.
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from config import lidar_params as cfg


# ============================================================
# 관측 결과 자료구조
# ============================================================

@dataclass
class RawScan:
    """rclpy LaserScan 콜백에서 받은 원시 스캔 스냅샷 (클러스터링 입력)."""
    ranges: np.ndarray       # 거리값 배열 (m)
    angle_min: float         # 첫 포인트 각도 (라디안)
    angle_increment: float   # 포인트간 각도 간격 (라디안)
    range_min: float         # 유효 최소거리 (m)
    range_max: float         # 유효 최대거리 (m)
    stamp: float             # 수신 시각 (time.monotonic())


@dataclass
class Cluster:
    """라이다 포인트들을 하나로 묶은 클러스터(물체 후보) 하나."""
    n: int              # 클러스터에 속한 포인트 개수 (개)
    dmin: float         # 클러스터 내 최소 거리 (m)
    dmed: float         # 클러스터 내 중간값 거리 (m)
    angle: float        # 클러스터 중심 각도 (도, +=좌측)
    angle_span: float   # 클러스터 각도 폭 (도) -- 콘/충돌형상 판정에 씀
    x_min: float        # 전방(x) 최소값 (m)
    x_max: float        # 전방(x) 최대값 (m)
    y_min: float        # 횡방향(y) 최소값 (m, +=좌측)
    y_max: float        # 횡방향(y) 최대값 (m)
    cx: float           # 전방(x) 중심 = (x_min+x_max)/2 (m)
    cy: float           # 횡방향(y) 중심 = (y_min+y_max)/2 (m)
    width: float        # 클러스터 폭 (bounding box 대각선 길이, m)


# ============================================================
# 1. SCAN_CAPTURE -- rclpy /scan 구독 (백그라운드 노드, 콜백은 저장만)
# ============================================================
# 콜백에서는 최신 스캔을 락 걸고 저장만 한다 -- 무거운 계산(클러스터링)은 절대
# 콜백 안에서 하지 않는다. capture_scan()이 호출되는 시점(메인 루프 프레임 주기)에
# 그때의 스냅샷을 복사해서 돌려준다.

_scan_lock = threading.Lock()
_latest_scan: RawScan | None = None
_node = None
_spin_thread = None
_started_rclpy = False


class _ScanSubscriberNode(Node):
    def __init__(self):
        super().__init__("physicar_lidar_scan")
        self.create_subscription(LaserScan, cfg.LIDAR_TOPIC, self._on_scan, qos_profile_sensor_data)

    def _on_scan(self, msg):
        global _latest_scan
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return
        scan = RawScan(
            ranges=ranges,
            angle_min=float(msg.angle_min),
            angle_increment=float(msg.angle_increment),
            range_min=float(msg.range_min),
            range_max=float(msg.range_max),
            stamp=time.monotonic(),
        )
        with _scan_lock:
            _latest_scan = scan


def _spin():
    try:
        rclpy.spin(_node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except Exception as e:
        print(f"[lidar] spin error: {type(e).__name__}: {e}")


def _ensure_started():
    """첫 capture_scan() 호출 시 rclpy 노드+스핀 스레드를 lazy하게 기동.
    이미 떠있으면 아무 것도 하지 않는다 (idempotent)."""
    global _node, _spin_thread, _started_rclpy

    if _node is not None:
        return

    if not rclpy.ok():
        rclpy.init(args=None)
        _started_rclpy = True

    _node = _ScanSubscriberNode()
    _spin_thread = threading.Thread(target=_spin, daemon=True, name="lidar-scan-spin")
    _spin_thread.start()
    print(f"[lidar] subscribed to {cfg.LIDAR_TOPIC}")


def capture_scan() -> RawScan | None:
    """최신 라이다 스캔 스냅샷을 반환. 아직 첫 스캔이 도착하기 전이면 None."""
    _ensure_started()
    with _scan_lock:
        return _latest_scan


def stop_scan():
    """구독 노드/스레드를 정리. main.py의 finally 블록에서 호출할 수 있게 제공하는
    함수 -- 지금은 아무 데서도 자동으로 부르지 않음(스핀 스레드가 데몬이라 프로세스
    종료 시 함께 죽긴 하지만, 명시적으로 정리하고 싶을 때 쓸 것)."""
    global _node

    try:
        if _node is not None:
            _node.destroy_node()
            _node = None
    except Exception:
        pass

    try:
        if _started_rclpy and rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

    if _spin_thread is not None and _spin_thread.is_alive():
        _spin_thread.join(timeout=1.0)

    print("[lidar] scan subscription stopped")


# ============================================================
# 2. CLUSTERING -- 인접한 포인트를 하나의 물체(클러스터)로 묶기
# ============================================================

def build_clusters(scan: RawScan | None) -> list[Cluster]:
    """원시 스캔을 인접 포인트끼리 묶어 클러스터 리스트로 변환.
    인접 조건: 배열상 바로 옆 인덱스(빠진 빔 없이 연속)이면서, XY 거리 차이가
    LIDAR_CLUSTER_LINK_M 이내. 가까운 클러스터부터 정렬해서 반환."""
    if scan is None:
        return []

    ranges = scan.ranges
    idx = np.arange(ranges.size, dtype=np.float32)
    angles_rad = scan.angle_min + idx * scan.angle_increment
    angles_deg = np.degrees(angles_rad)

    valid = (
        np.isfinite(ranges)
        & (ranges >= scan.range_min)
        & (ranges <= min(scan.range_max, cfg.LIDAR_CLUSTER_MAX_RANGE_M))
        & (angles_deg >= -cfg.LIDAR_CLUSTER_FOV_DEG)
        & (angles_deg <= cfg.LIDAR_CLUSTER_FOV_DEG)
    )
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return []

    xs = ranges * np.cos(angles_rad)
    ys = ranges * np.sin(angles_rad)

    clusters_idx = []
    current = [int(valid_idx[0])]
    prev_i = int(valid_idx[0])

    for raw_i in valid_idx[1:]:
        i = int(raw_i)
        index_gap = i - prev_i
        xy_gap = math.hypot(float(xs[i] - xs[prev_i]), float(ys[i] - ys[prev_i]))

        if index_gap == 1 and xy_gap <= cfg.LIDAR_CLUSTER_LINK_M:
            current.append(i)
        else:
            if len(current) >= cfg.LIDAR_CLUSTER_MIN_POINTS:
                clusters_idx.append(current)
            current = [i]   # 간격이 벌어졌으면 새 클러스터 시작 (빠진 빔 사이는 잇지 않음)

        prev_i = i

    if len(current) >= cfg.LIDAR_CLUSTER_MIN_POINTS:
        clusters_idx.append(current)

    clusters = []
    for inds in clusters_idx:
        inds = np.asarray(inds, dtype=np.int32)
        rr, aa = ranges[inds], angles_deg[inds]
        xx, yy = xs[inds], ys[inds]

        x_min, x_max = float(np.min(xx)), float(np.max(xx))
        y_min, y_max = float(np.min(yy)), float(np.max(yy))
        width = float(math.hypot(x_max - x_min, y_max - y_min))

        clusters.append(Cluster(
            n=int(inds.size),
            dmin=float(np.min(rr)),
            dmed=float(np.median(rr)),
            angle=float(np.median(aa)),
            angle_span=float(np.max(aa) - np.min(aa)),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            cx=0.5 * (x_min + x_max), cy=0.5 * (y_min + y_max),
            width=width,
        ))

    clusters.sort(key=lambda c: c.dmin)   # 가까운 클러스터부터
    return clusters


# ============================================================
# 3. CONE_CLASSIFICATION -- 클러스터 중 콘(라바콘) 형상에 맞는 것만 골라내기
# ============================================================
# 매 프레임 독립적으로 "이 클러스터가 콘처럼 생겼나"만 판정하는 stateless 필터.
# "같은 콘을 계속 추적 중인가"(여러 프레임에 걸친 락/추적)는 여기서 하지 않는다 --
# 그건 control/obstacle_avoid.py가 프레임간 상태로 들고 있을 몫.

def _is_cone_shape(c: Cluster, corner_hint: bool) -> bool:
    """포인트 수/폭/각도폭이 콘 하나로 볼만한 형태인지."""
    if corner_hint:
        return (
            c.n >= cfg.CONE_CORNER_MIN_POINTS
            and cfg.CONE_CORNER_WIDTH_MIN_M <= c.width <= cfg.CONE_CORNER_WIDTH_MAX_M
            and c.angle_span <= cfg.CONE_CORNER_MAX_ANGLE_SPAN_DEG
        )
    return (
        c.n >= cfg.CONE_MIN_POINTS
        and cfg.CONE_WIDTH_MIN_M <= c.width <= cfg.CONE_WIDTH_MAX_M
        and c.angle_span <= cfg.CONE_MAX_ANGLE_SPAN_DEG
    )


def classify_cone_candidates(clusters: list[Cluster], corner_hint: bool = False) -> list[Cluster]:
    """클러스터 중 콘(라바콘) 형상 + 탐색범위(거리/각도)에 맞는 것만 골라냄.
    corner_hint=True면 코너 구간용으로 더 넓은 탐색 게이트를 쓴다 -- 코너에서는 콘이
    더 큰 각도에서 보이고, 비스듬히 보여서 폭 측정도 더 흔들리기 때문."""
    if corner_hint:
        max_range = cfg.CONE_CORNER_DETECT_MAX_RANGE_M
        search_angle = cfg.CONE_CORNER_SEARCH_ANGLE_DEG
        max_abs_y = cfg.CONE_CORNER_SEARCH_MAX_ABS_Y_M
        min_forward_x = 0.15   # 이보다 차량에 가까우면 콘이 아니라 차체/노이즈로 취급
    else:
        max_range = cfg.CONE_DETECT_MAX_RANGE_M
        search_angle = cfg.CONE_SEARCH_ANGLE_DEG
        max_abs_y = cfg.CONE_SEARCH_MAX_ABS_Y_M
        min_forward_x = 0.25

    return [
        c for c in clusters
        if _is_cone_shape(c, corner_hint)
        and c.dmin <= max_range
        and abs(c.angle) <= search_angle
        and c.x_max > min_forward_x
        and abs(c.cy) <= max_abs_y
    ]


# ============================================================
# 4. DEBUG VISUALIZATION -- 클러스터를 위에서 내려다본 2D 산점도로 표시
# ============================================================

_DEBUG_CANVAS_PX = 520          # 캔버스 한 변 크기 (px)
_DEBUG_SCALE_PX_PER_M = 160.0   # 1m를 몇 px로 그릴지


def draw_debug(clusters: list[Cluster]) -> np.ndarray:
    """클러스터를 위에서 내려다본(전방=위쪽) 2D 산점도로 렌더링.
    콘 후보(직선 탐색 게이트 기준)는 노란색, 그 외 클러스터는 회색으로 표시."""
    size = _DEBUG_CANVAS_PX
    img = np.zeros((size, size, 3), dtype=np.uint8)
    origin = (size // 2, size - 30)   # 차량 위치: 캔버스 하단 중앙

    for r_m in np.arange(0.5, cfg.LIDAR_CLUSTER_MAX_RANGE_M + 0.01, 0.5):
        cv2.circle(img, origin, int(r_m * _DEBUG_SCALE_PX_PER_M), (40, 40, 40), 1)

    fov = math.radians(cfg.LIDAR_CLUSTER_FOV_DEG)
    far_px = cfg.LIDAR_CLUSTER_MAX_RANGE_M * _DEBUG_SCALE_PX_PER_M
    for sign in (-1, 1):
        end = (int(origin[0] + sign * far_px * math.sin(fov)), int(origin[1] - far_px * math.cos(fov)))
        cv2.line(img, origin, end, (40, 40, 40), 1)

    cone_ids = {id(c) for c in classify_cone_candidates(clusters, corner_hint=False)}

    for c in clusters:
        # 화면: cx(전방)=위쪽, cy(+좌측)=왼쪽. y축은 이미지 아래가 +이므로 부호 반전.
        px = int(origin[0] - c.cy * _DEBUG_SCALE_PX_PER_M)
        py = int(origin[1] - c.cx * _DEBUG_SCALE_PX_PER_M)
        color = (0, 255, 255) if id(c) in cone_ids else (150, 150, 150)
        cv2.circle(img, (px, py), 5, color, -1)
        cv2.putText(img, f"{c.dmin:.2f}m", (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.circle(img, origin, 6, (255, 255, 255), -1)   # 차량 위치
    cv2.putText(img, f"clusters={len(clusters)}  cone_candidates={len(cone_ids)}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ============================================================
# 단독 실행: 인식 결과(클러스터/콘 후보)만 웹 디버그 뷰로 확인
# ============================================================
if __name__ == "__main__":
    from config import base as base_cfg
    from myapp import debug_view

    debug_view.serve()
    print("[lidar.py] perception-only self test -- http://localhost:5000")

    try:
        while True:
            t0 = time.time()
            scan = capture_scan()
            clusters = build_clusters(scan)
            cone_candidates = classify_cone_candidates(clusters, corner_hint=False)
            panel = draw_debug(clusters)

            status = f"clusters={len(clusters)} cone_candidates={len(cone_candidates)}"
            debug_view.update_web(panel, status)

            proc_time = time.time() - t0
            time.sleep(max(0.0, 1.0 / base_cfg.TARGET_FPS - proc_time))
    except KeyboardInterrupt:
        print("\n[lidar.py] stopped")
    finally:
        stop_scan()
        debug_view.stop_view()
