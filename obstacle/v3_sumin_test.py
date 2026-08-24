
import math
import time
import config

# -----------------------------------------------------------------------------
# 튜닝값
# -----------------------------------------------------------------------------
DETECT_RANGE_M = 1.60
FRONT_HALF_ANGLE_DEG = 60.0
CLUSTER_GAP_DEG = 6.0
MIN_PASSABLE_GAP_DEG = 18.0
EDGE_TARGET_CLEARANCE_M = 0.18
GAP_STEER_GAIN = 1.0
PROXIMITY_FLOOR = 0.35
BIAS_MAX_DEG = 12.0
GAP_TARGET_ALPHA = 0.50
BIAS_RATE_LIMIT_DEG = 6.0
STICKY_SLACK_DEG = 15.0
STICKY_CLEAR_FRAMES = 3
DEBUG_PRINT_SEC = 0.50


def _front_points(points):
    """전방 ±60°, 1.6m 안쪽 LiDAR만 사용. angle 부호는 그대로 유지."""
    front = [
        (float(a), float(d))
        for a, d in points
        if abs(float(a)) <= FRONT_HALF_ANGLE_DEG
        and 0.02 < float(d) < DETECT_RANGE_M
    ]
    front.sort(key=lambda p: p[0])
    return front


def _cluster_front_points(front_sorted):
    """연속된 각도 점들을 하나의 장애물 cluster로 묶음."""
    if not front_sorted:
        return []

    raw_clusters = []
    current = [front_sorted[0]]

    for a, d in front_sorted[1:]:
        if a - current[-1][0] > CLUSTER_GAP_DEG:
            raw_clusters.append(current)
            current = []
        current.append((a, d))

    raw_clusters.append(current)

    clusters = []
    for c in raw_clusters:
        clusters.append({
            "start": min(a for a, _ in c),
            "end": max(a for a, _ in c),
            "min_dist": min(d for _, d in c),
            "n": len(c),
        })
    return clusters


def _find_gaps(clusters):
    """cluster 사이 + 시야 양끝의 빈 각도 구간을 모두 반환."""
    if not clusters:
        return []

    gaps = []
    n = len(clusters)
    for i in range(n + 1):
        start = -FRONT_HALF_ANGLE_DEG if i == 0 else clusters[i - 1]["end"]
        end = FRONT_HALF_ANGLE_DEG if i == n else clusters[i]["start"]
        width = end - start
        if width > 0:
            gaps.append({
                "start": float(start),
                "end": float(end),
                "width": float(width),
                "mid": float((start + end) / 2.0),
                "interior": bool(i != 0 and i != n),
            })
    return gaps


def _edge_gap_target(gap, obstacle_dist):
    """가장자리 gap 중앙이 아니라 콘 경계에서 0.18m 정도만 벗어난 각도 목표."""
    margin_deg = math.degrees(
        math.atan2(EDGE_TARGET_CLEARANCE_M, max(float(obstacle_dist), 0.05))
    )
    start = gap["start"]
    end = gap["end"]

    # +60도 쪽 시야끝에 닿으면 왼쪽이 열려 있음.
    if end >= FRONT_HALF_ANGLE_DEG - 1e-6:
        return min(end, start + margin_deg)

    # -60도 쪽 시야끝에 닿으면 오른쪽이 열려 있음.
    return max(start, end - margin_deg)


def _same_sign(value, sign_hint):
    if sign_hint == 0.0 or value == 0.0:
        return False
    return (value > 0.0) == (sign_hint > 0.0)


def _choose_gap_target(clusters, gaps, curve_direction, sticky_sign):
    """내부 gap 우선, 없으면 가장자리 gap을 sticky 방향과 함께 선택."""
    if not gaps:
        if not clusters:
            return None, 999.0
        center = 0.5 * (clusters[0]["start"] + clusters[-1]["end"])
        target = -FRONT_HALF_ANGLE_DEG if center >= 0.0 else FRONT_HALF_ANGLE_DEG
        return float(target), 0.0

    interior = [
        g for g in gaps
        if g["interior"] and g["width"] >= MIN_PASSABLE_GAP_DEG
    ]
    edge = [g for g in gaps if not g["interior"]]

    # 두 콘 사이 충분한 내부 틈이 있으면 그 중앙을 사용.
    if interior:
        best = max(interior, key=lambda g: g["width"])
        return float(best["mid"]), float(best["width"])

    if edge:
        widest = max(edge, key=lambda g: g["width"])
        best = widest

        # 한 번 고른 방향을 우선. 아직 방향이 없을 때만 lane curve hint 사용.
        hint = sticky_sign if sticky_sign != 0.0 else (curve_direction or 0.0)
        if hint:
            aligned = [g for g in edge if _same_sign(g["mid"], hint)]
            if aligned:
                best_aligned = max(aligned, key=lambda g: g["width"])
                # sticky 쪽이 확실히 더 막힌 경우에만 반대쪽으로 전환.
                if best_aligned["width"] >= widest["width"] - STICKY_SLACK_DEG:
                    best = best_aligned

        obstacle_dist = min(c["min_dist"] for c in clusters)
        target = _edge_gap_target(best, obstacle_dist)
        return float(target), float(best["width"])

    # 내부 gap이 모두 너무 좁고 edge도 없으면 덜 막힌 방향으로 최대한 붙음.
    center = 0.5 * (clusters[0]["start"] + clusters[-1]["end"])
    target = -FRONT_HALF_ANGLE_DEG if center >= 0.0 else FRONT_HALF_ANGLE_DEG
    return float(target), 0.0


class ObstacleAvoider:
    """현재 main.py와 동일한 step() 인터페이스를 유지하는 gap-following 회피 모듈."""

    def __init__(self):
        self.last_valid_points = []
        self.last_ok_time = time.time()
        self.smoothed_target_angle = 0.0
        self.prev_bias = 0.0
        self.sticky_sign = 0.0  # +1 left / -1 right / 0 none
        self.clear_frames = 0
        self.last_debug_time = 0.0

    def _clear_state(self):
        self.clear_frames += 1
        if self.clear_frames >= STICKY_CLEAR_FRAMES:
            self.sticky_sign = 0.0
            self.smoothed_target_angle = 0.0
            self.prev_bias = 0.0

    def _debug(self, now, clusters, target_angle, gap_width, min_dist, bias, state):
        if now - self.last_debug_time < DEBUG_PRINT_SEC:
            return
        self.last_debug_time = now

        cluster_text = ", ".join(
            f"[{c['start']:+.0f}~{c['end']:+.0f}deg/{c['min_dist']:.2f}m]"
            for c in clusters
        ) or "none"
        target_text = "----" if target_angle is None else f"{target_angle:+.1f}deg"
        dist_text = "----" if not math.isfinite(min_dist) else f"{min_dist:.2f}m"
        sticky_text = "L" if self.sticky_sign > 0 else "R" if self.sticky_sign < 0 else "-"

        print(
            f"[LIDAR GAP] {state:<15} dist={dist_text:<6} "
            f"target={target_text:<9} gap={gap_width:5.1f}deg "
            f"bias={bias:+5.1f}deg sticky={sticky_text} clusters={cluster_text}"
        )

    def step(self, points, curve_direction, corner_active, now):
        """한 프레임의 obstacle bias 계산. main.py 호환 dict 반환."""

        # 짧은 LiDAR 끊김은 마지막 유효 스캔 재사용.
        if points:
            self.last_valid_points = points
            self.last_ok_time = now

        stale_for = now - self.last_ok_time
        if points:
            points_for_avoidance = points
        elif stale_for <= config.LIDAR_STALE_GRACE_S:
            points_for_avoidance = self.last_valid_points
        else:
            points_for_avoidance = []

        front = _front_points(points_for_avoidance)

        if not front:
            self._clear_state()
            self._debug(now, [], None, 999.0, math.inf, 0.0, "CLEAR")
            return {
                "bias": 0.0,
                "min_dist": math.inf,
                "direction": 0.0,
                "boxed_in": False,
                "backing": False,
                "backup_steer": 0.0,
                "backup_speed": getattr(config, "BACKUP_SPEED", -0.20),
                "backup_attempts": 0,
                "stopped_boxed": False,
                "lidar_stale_for": stale_for,
                "state": "CLEAR",
                "target_angle": None,
                "gap_width": 999.0,
                "clusters": [],
            }

        self.clear_frames = 0
        min_dist = min(d for _, d in front)
        clusters = _cluster_front_points(front)
        gaps = _find_gaps(clusters)
        target_angle, gap_width = _choose_gap_target(
            clusters, gaps, curve_direction, self.sticky_sign
        )

        # target 방향 latch.
        if target_angle is not None and abs(target_angle) >= 1.0:
            target_sign = 1.0 if target_angle > 0.0 else -1.0
            if self.sticky_sign == 0.0 or target_sign != self.sticky_sign:
                self.sticky_sign = target_sign

        # target angle EMA + 거리 기반 회피 세기.
        if target_angle is None:
            self.smoothed_target_angle = 0.0
            desired_bias = 0.0
        else:
            self.smoothed_target_angle = (
                GAP_TARGET_ALPHA * float(target_angle)
                + (1.0 - GAP_TARGET_ALPHA) * self.smoothed_target_angle
            )
            proximity = PROXIMITY_FLOOR + (1.0 - PROXIMITY_FLOOR) * (
                1.0 - min(max(min_dist / DETECT_RANGE_M, 0.0), 1.0)
            )
            desired_bias = GAP_STEER_GAIN * self.smoothed_target_angle * proximity
            desired_bias = max(-BIAS_MAX_DEG, min(BIAS_MAX_DEG, desired_bias))

        # 3_1 철학: 코너에서는 멀리 있는 LiDAR가 기본 코너 조향을 망가뜨리지 않음.
        corner_emergency_dist = getattr(config, "CORNER_OBSTACLE_EMERGENCY_DIST", 0.55)
        corner_obstacle_steer = getattr(config, "CORNER_OBSTACLE_STEER", False)
        if corner_active and not corner_obstacle_steer and min_dist >= corner_emergency_dist:
            desired_bias = 0.0
            state = "CORNER_SUPPRESS"
        else:
            state = "AVOID"

        # bias 급반전 방지.
        delta = desired_bias - self.prev_bias
        delta = max(-BIAS_RATE_LIMIT_DEG, min(BIAS_RATE_LIMIT_DEG, delta))
        bias = self.prev_bias + delta
        self.prev_bias = bias

        direction = 1.0 if bias > 0.5 else -1.0 if bias < -0.5 else 0.0
        self._debug(now, clusters, target_angle, gap_width, min_dist, bias, state)

        # 현재 main.py가 요구하는 모든 key 유지.
        # 첫 버전은 backing / stopped_boxed를 의도적으로 사용하지 않음.
        return {
            "bias": float(bias),
            "min_dist": float(min_dist),
            "direction": float(direction),
            "boxed_in": False,
            "backing": False,
            "backup_steer": 0.0,
            "backup_speed": getattr(config, "BACKUP_SPEED", -0.20),
            "backup_attempts": 0,
            "stopped_boxed": False,
            "lidar_stale_for": float(stale_for),
            # main.py는 무시하지만 로그/후속 디버깅용으로 남김.
            "state": state,
            "target_angle": None if target_angle is None else float(target_angle),
            "gap_width": float(gap_width),
            "clusters": [
                (float(c["start"]), float(c["end"]), float(c["min_dist"]), int(c["n"]))
                for c in clusters
            ],
        }
