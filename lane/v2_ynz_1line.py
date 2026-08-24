"""라인트레이싱 v4_ynz_1line -- 동적 차선 변경(Left/Right Lane) + 카메라 팬/틸트 단축 하이브리드.

이 모듈은 기존 v1_basic.py의 차선 감지 알고리즘을 개량하여,
1. 임포트 시 신호등 카메라 탐색 범위를 단축 (config.TRAFFIC_SEARCH_POSES 재정의)
2. 동적 차선 변경 주행:
   기본 주행은 왼쪽 차선(Left Lane, 중앙선과 왼쪽 흰선 사이)을 추적하며 달리고, 
   장애물 감지 시 오른쪽 차선(Right Lane, 중앙선과 오른쪽 흰선 사이)으로 경로를 변경하여 주행합니다.
   차선 폭의 1/4 지점을 목표 중심으로 삼아 안정적으로 편향 주행합니다.
3. 초기 차선 폭을 실제 트랙 해상도 비율(0.56 * w)에 맞춰 현실적으로 세팅하여 
   초기 우측 편향 및 벽 충돌/후진 문제를 원천 차단합니다.
"""
import math
import cv2
import numpy as np

import config

# ============================================================
# 카메라 탐색 기동 최소화 오버라이드 (팬/틸트 범위 및 횟수 단축)
# ============================================================
config.TRAFFIC_SEARCH_POSES = [
    (0.0, 0.0),
    (-20.0, 0.0),
    (20.0, 0.0),
]


# ============================================================
# BEV 변환 + 흰색 마스크
# ============================================================

def roi_points(w, h):
    pts = config.ROI_NORM.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.float32)


def build_bev_matrices(w, h):
    src = roi_points(w, h)
    margin = 0.15 * w
    dst = np.float32([
        [margin, 0], [w - margin, 0], [w - margin, h - 1], [margin, h - 1],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return src, M


def make_white_mask(bev_bgr):
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, config.WHITE_V_MIN], dtype=np.uint8)
    upper = np.array([179, config.WHITE_S_MAX, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# ============================================================
# 흰색 양쪽 경계선 슬라이딩 윈도우 추적
# ============================================================

def _search_base(histogram, x0, x1):
    x0, x1 = max(0, int(x0)), min(len(histogram), int(x1))
    if x1 <= x0:
        return None
    section = histogram[x0:x1]
    if section.size == 0 or np.max(section) <= 0:
        return None
    return int(x0 + np.argmax(section))


def sliding_window_fit(binary, side):
    """지정한 쪽(left/right)에서 흰색 경계선을 x = f(y)로 피팅. 반환: (polyfit 계수, 픽셀 수)."""
    h, w = binary.shape
    histogram = np.sum(binary[int(h * 0.45):, :] > 0, axis=0)
    a, b = config.LEFT_SEARCH if side == "left" else config.RIGHT_SEARCH
    base = _search_base(histogram, a * w, b * w)
    if base is None:
        return None, 0

    nonzero_y, nonzero_x = binary.nonzero()
    window_h = max(1, h // config.NWINDOWS)
    margin = max(10, int(w * config.WINDOW_MARGIN_RATIO))
    x_current = base
    lane_inds = []

    for win in range(config.NWINDOWS):
        y_low, y_high = h - (win + 1) * window_h, h - win * window_h
        x_low, x_high = x_current - margin, x_current + margin
        good = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high))
        inds = np.where(good)[0]
        if inds.size:
            lane_inds.append(inds)
        if inds.size >= config.MINPIX:
            x_current = int(np.mean(nonzero_x[inds]))

    if not lane_inds:
        return None, 0
    lane_inds = np.concatenate(lane_inds)
    xs, ys = nonzero_x[lane_inds], nonzero_y[lane_inds]
    if len(xs) < config.MIN_FIT_PIXELS:
        return None, len(xs)
    return np.polyfit(ys, xs, 2), len(xs)


def fit_x(fit, y):
    if fit is None:
        return None
    return float(fit[0] * y * y + fit[1] * y + fit[2])


class LaneTracker:
    """동적 차선(왼쪽 차선 / 오른쪽 차선) 추적기.
    
    config.target_lane 값("LEFT" 또는 "RIGHT")에 따라 목표 경로 중심을 변경합니다:
    - LEFT Lane: 왼쪽 선 + lane_width_px / 4
    - RIGHT Lane: 오른쪽 선 - lane_width_px / 4
    """

    def __init__(self):
        self.lane_width_px = None

    def detect(self, mask):
        h, w = mask.shape
        if self.lane_width_px is None:
            self.lane_width_px = 0.56 * w  # 초기 차선 폭을 크게 설정하여 우측 쏠림 방지

        y_near, y_far = int(h * config.NEAR_Y_RATIO), int(h * config.FAR_Y_RATIO)

        left_fit, left_count = sliding_window_fit(mask, "left")
        right_fit, right_count = sliding_window_fit(mask, "right")

        left_near = fit_x(left_fit, y_near)
        left_far = fit_x(left_fit, y_far)
        right_near = fit_x(right_fit, y_near)
        right_far = fit_x(right_fit, y_far)

        both_ok = all(v is not None for v in [left_near, right_near, left_far, right_far])
        if both_ok:
            min_w, max_w = w * config.LANE_WIDTH_MIN_RATIO, w * config.LANE_WIDTH_MAX_RATIO
            if (min_w <= right_near - left_near <= max_w and
                    min_w <= right_far - left_far <= max_w and
                    left_near < right_near and left_far < right_far):
                observed = 0.5 * ((right_near - left_near) + (right_far - left_far))
                self.lane_width_px = (config.LANE_WIDTH_ALPHA * observed +
                                       (1 - config.LANE_WIDTH_ALPHA) * self.lane_width_px)

        # 동적 차선 타겟 획득 (ObstacleAvoider 가 설정하는 config.target_lane 참조)
        target_lane = getattr(config, "target_lane", "LEFT")

        if target_lane == "LEFT":
            # 왼쪽 차선 주행 모드
            if left_near is not None and left_far is not None:
                center_near = left_near + self.lane_width_px / 4.0
                center_far = left_far + self.lane_width_px / 4.0
                mode, confidence = "LEFT_ONLY", 0.80
            elif right_near is not None and right_far is not None:
                # 왼쪽 선이 안 보이면 오른쪽 선 기준 역추정
                center_near = right_near - 3.0 * self.lane_width_px / 4.0
                center_far = right_far - 3.0 * self.lane_width_px / 4.0
                mode, confidence = "RIGHT_ONLY", 0.60
            else:
                center_near = center_far = None
                mode, confidence = "LOST", 0.0
        else:
            # 오른쪽 차선 주행 모드
            if right_near is not None and right_far is not None:
                center_near = right_near - self.lane_width_px / 4.0
                center_far = right_far - self.lane_width_px / 4.0
                mode, confidence = "RIGHT_ONLY", 0.80
            elif left_near is not None and left_far is not None:
                # 오른쪽 선이 안 보이면 왼쪽 선 기준 역추정
                center_near = left_near + 3.0 * self.lane_width_px / 4.0
                center_far = left_far + 3.0 * self.lane_width_px / 4.0
                mode, confidence = "LEFT_ONLY", 0.60
            else:
                center_near = center_far = None
                mode, confidence = "LOST", 0.0

        if center_near is not None and not (-0.15 * w <= center_near <= 1.15 * w):
            center_near = center_far = None
            mode, confidence = "LOST", 0.0

        return {
            "left_fit": left_fit, "right_fit": right_fit,
            "left_count": left_count, "right_count": right_count,
            "center_near": center_near, "center_far": center_far,
            "confidence": confidence, "mode": mode,
            "lane_width_px": self.lane_width_px,
            "y_near": y_near, "y_far": y_far,
        }


def lane_center_at_y(det, y):
    """동적 차선 타겟에 근거한 중심선 추출"""
    left = fit_x(det["left_fit"], y)
    right = fit_x(det["right_fit"], y)
    width = det.get("lane_width_px")
    target_lane = getattr(config, "target_lane", "LEFT")

    if target_lane == "LEFT":
        if left is not None:
            return left + width / 4.0
        if right is not None:
            return right - 3.0 * width / 4.0
    else:
        if right is not None:
            return right - width / 4.0
        if left is not None:
            return left + 3.0 * width / 4.0
    return None


# ============================================================
# Stanley 조향
# ============================================================

def steering_from_lane(det, w, current_speed):
    center_near, center_far = det["center_near"], det["center_far"]
    if center_near is None or center_far is None:
        return None, None, None
    lateral_error = (center_near - w / 2.0) / (w / 2.0)
    heading_error = (center_far - center_near) / (w / 2.0)

    cross_track_term = math.atan(
        (config.K_LATERAL * 0.1 * lateral_error) /
        (max(0.05, current_speed) + config.STANLEY_SOFT_FACTOR))
    cross_track_deg = math.degrees(cross_track_term)
    heading_deg = config.K_HEADING * 0.1 * heading_error * config.STANLEY_HEADING_SCALE

    steering = -(heading_deg + cross_track_deg * config.STANLEY_CROSS_TRACK_SCALE)   # + = 좌
    if abs(steering) < config.STEER_DEADBAND_DEG:
        steering = 0.0
    steering = float(np.clip(steering, -config.STEER_MAX, config.STEER_MAX))
    return steering, lateral_error, heading_error


# ============================================================
# 속도 계획 (미리보기 곡률)
# ============================================================

def preview_lane_geometry(det, h, w, current_speed):
    speed_ratio = float(np.clip(current_speed / max(config.SPEED_MAX, 1e-6), 0.0, 1.0))
    far_ratio = float(np.clip(0.55 - 0.10 * speed_ratio, config.PREVIEW_MIN_Y_RATIO, 0.60))
    ratios = np.linspace(0.82, far_ratio, 5, dtype=np.float32)

    centers = []
    for r in ratios:
        y = int(h * float(r))
        x = lane_center_at_y(det, y)
        if x is not None and -0.15 * w <= x <= 1.15 * w:
            centers.append((y, x))
    if len(centers) < 3:
        return None, 0.0, centers

    y_near, x_near = centers[0]
    y_far, x_far = centers[-1]
    preview_heading = (x_far - x_near) / (w / 2.0)

    slopes = []
    for i in range(len(centers) - 1):
        y0, x0 = centers[i]
        y1, x1 = centers[i + 1]
        dy = max(1.0, float(y0 - y1))
        slopes.append((x1 - x0) / dy)

    if len(slopes) >= 2:
        curvature_from_slope = float(np.clip(
            float(np.mean(np.abs(np.diff(np.asarray(slopes, dtype=np.float32))))) * 2.0, 0.0, 1.0))
    else:
        curvature_from_slope = 0.0
    heading_level = float(np.clip(abs(preview_heading) / 0.30, 0.0, 1.0))
    curvature_level = max(curvature_from_slope, heading_level * 0.65)
    return float(preview_heading), float(curvature_level), centers


def choose_target_speed(det, steering, heading_error, preview_curvature=0.0):
    if steering is None or heading_error is None:
        return 0.0
    if det["mode"] == "LOST":
        return 0.0
    if det["mode"] in ("LEFT_ONLY", "RIGHT_ONLY"):
        return config.SPEED_ONE_LINE

    s, h_, p = abs(float(steering)), abs(float(heading_error)), float(preview_curvature)
    current_curve = max(np.clip(h_ / 0.22, 0.0, 1.0), np.clip(s / 18.0, 0.0, 1.0))
    combined_curve = max(p * 0.85, current_curve * 0.55)

    if combined_curve <= config.STRAIGHT_CURVE_LIMIT:
        return config.SPEED_MAX

    effective_curve = np.clip(
        (combined_curve - config.CURVE_DEADBAND) / max(1.0 - config.CURVE_DEADBAND, 1e-6), 0.0, 1.0)
    speed = config.SPEED_MAX - effective_curve * (config.SPEED_MAX - config.SPEED_MIN)
    return float(np.clip(speed, config.SPEED_MIN, config.SPEED_MAX))


def ramp_speed(current, target):
    if target > current:
        return min(target, current + config.SPEED_RAMP_UP)
    return max(target, current - config.SPEED_RAMP_DOWN)


# ============================================================
# 코너 폴백: 노란 점선 경로 추적
# ============================================================

def make_yellow_corner_mask(bev_bgr):
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array([config.YELLOW_H_MIN, config.YELLOW_S_MIN, config.YELLOW_V_MIN], dtype=np.uint8),
        np.array([config.YELLOW_H_MAX, 255, 255], dtype=np.uint8))
    v = hsv[:, :, 2]
    asphalt = cv2.dilate(cv2.inRange(v, 0, config.ASPHALT_V_MAX),
                          np.ones((config.ASPHALT_DILATE, config.ASPHALT_DILATE), np.uint8))
    yellow = cv2.bitwise_and(yellow, asphalt)
    k3 = np.ones((3, 3), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k3)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k3)
    return yellow


def yellow_centroids(yellow_mask):
    h, w = yellow_mask.shape
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (config.YELLOW_MIN_AREA <= area <= config.YELLOW_MAX_AREA):
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        if not (0.03 * w <= cx <= 0.97 * w and 0.05 * h <= cy <= 0.98 * h):
            continue
        pts.append((cx, cy, area))
    return pts


def build_corner_path(points, w, h):
    if not points:
        return []
    start = np.array([w / 2.0, h - 8.0], dtype=np.float32)
    unused = list(points)

    dists = [np.linalg.norm(np.array([p[0], p[1]], dtype=np.float32) - start) for p in unused]
    first_i = int(np.argmin(dists))
    if dists[first_i] > config.CORNER_FIRST_LINK_PX:
        return []
    first = unused.pop(first_i)
    path = [(first[0], first[1])]
    current = np.array([first[0], first[1]], dtype=np.float32)

    for _ in range(14):
        if not unused:
            break
        best_i, best_cost = None, None
        for i, p in enumerate(unused):
            q = np.array([p[0], p[1]], dtype=np.float32)
            dist = float(np.linalg.norm(q - current))
            if dist > config.CORNER_MAX_LINK_PX:
                continue
            backward = max(0.0, float(p[1] - current[1] - config.CORNER_BACKWARD_ALLOW_PX))
            cost = dist + 3.0 * backward
            if best_cost is None or cost < best_cost:
                best_cost, best_i = cost, i
        if best_i is None:
            break
        p = unused.pop(best_i)
        path.append((p[0], p[1]))
        current = np.array([p[0], p[1]], dtype=np.float32)
    return path


def choose_corner_target(path, w, h):
    if not path:
        return None
    start = np.array([w / 2.0, h - 8.0], dtype=np.float32)
    prev, travelled = start, 0.0
    target = np.array(path[-1], dtype=np.float32)
    for p in path:
        q = np.array(p, dtype=np.float32)
        travelled += float(np.linalg.norm(q - prev))
        target = q
        if travelled >= config.CORNER_LOOKAHEAD_PX:
            break
        prev = q
    return float(target[0]), float(target[1])


def steering_from_corner_target(target, w, h):
    if target is None:
        return None
    vehicle_x, vehicle_y = w / 2.0, h - 8.0
    dx = float(target[0] - vehicle_x)
    forward = max(12.0, float(vehicle_y - target[1]))
    target_angle_deg = math.degrees(math.atan2(dx, forward))
    steering = -config.CORNER_STEER_GAIN * target_angle_deg    # + = 좌
    return float(np.clip(steering, -config.STEER_MAX, config.STEER_MAX))


# ============================================================
# 학습된 CNN 폴백 (하이브리드)
# ============================================================
_lane_session = None


def lane_model_steer(img):
    global _lane_session
    if _lane_session is None:
        import onnxruntime as ort
        _lane_session = ort.InferenceSession(config.LANE_MODEL_PATH, providers=["CPUExecutionProvider"])
    x = cv2.resize(img, (config.LANE_MODEL_WIDTH, config.LANE_MODEL_HEIGHT))
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    steer = float(_lane_session.run(None, {"camera": x})[0][0][0])
    return max(-config.STEER_MAX, min(config.STEER_MAX, steer))


# ============================================================
# LaneKeeper -- 상태(코너모드/조향 EMA/속도 램프)를 들고 있는 제어기
# ============================================================

class LaneKeeper:
    def __init__(self):
        self.tracker = LaneTracker()
        self.corner_active = False
        self.corner_exit_count = 0
        self.corner_hold_count = 0
        self.weak_white_count = 0
        self.sharp_count = 0
        self.smoothed_lateral_error = 0.0
        self.smoothed_center_near = None
        self.smoothed_center_far = None
        self.last_corner_steer = 0.0
        self.smoothed_steer = 0.0
        self.current_speed = 0.0
        self.smoothed_target_speed = None
        self.last_steer_sign = 1.0
        self.last_debug = {}

    def step(self, img):
        if img is None:
            return self._lost_step("no camera frame")

        if config.USE_LEARNED_LANE_MODEL:
            try:
                steer = lane_model_steer(img)
                self.current_speed = ramp_speed(self.current_speed, config.SPEED_MAX * 0.7)
                if abs(steer) > 1.0:
                    self.last_steer_sign = 1.0 if steer > 0 else -1.0
                curve_direction = (1.0 if steer > 0 else -1.0) if abs(steer) >= config.LINE_CURVE_THRESHOLD else None
                return steer, self.current_speed, curve_direction, f"learned steer {steer:+.1f}"
            except Exception as e:
                print(f"lane model error: {e} -- BEV 파이프라인으로 폴백")

        h, w = img.shape[:2]
        src_pts, M = build_bev_matrices(w, h)
        bev = cv2.warpPerspective(img, M, (w, h))

        white_mask = make_white_mask(bev)
        white_mask[:int(h * 0.18), :] = 0

        det = self.tracker.detect(white_mask)

        det_for_steer = det
        if det["center_near"] is not None and det["center_far"] is not None:
            a = config.CENTER_SMOOTH_ALPHA
            self.smoothed_center_near = det["center_near"] if self.smoothed_center_near is None else (
                a * det["center_near"] + (1.0 - a) * self.smoothed_center_near)
            self.smoothed_center_far = det["center_far"] if self.smoothed_center_far is None else (
                a * det["center_far"] + (1.0 - a) * self.smoothed_center_far)
            det_for_steer = dict(det)
            det_for_steer["center_near"] = self.smoothed_center_near
            det_for_steer["center_far"] = self.smoothed_center_far
        else:
            self.smoothed_center_near = self.smoothed_center_far = None

        steering, lateral_error, heading_error = steering_from_lane(det_for_steer, w, self.current_speed)
        if lateral_error is not None:
            self.smoothed_lateral_error = (config.LANE_BIAS_ALPHA * lateral_error +
                                            (1.0 - config.LANE_BIAS_ALPHA) * self.smoothed_lateral_error)

        preview_heading, preview_curvature, _ = preview_lane_geometry(det, h, w, self.current_speed)

        yellow_mask = make_yellow_corner_mask(bev)
        yellow_pts = yellow_centroids(yellow_mask)
        corner_path = build_corner_path(yellow_pts, w, h)
        corner_target = choose_corner_target(corner_path, w, h)
        corner_steering = steering_from_corner_target(corner_target, w, h)

        # 코너 진입 디바운스
        normal_sharp_now = heading_error is not None and abs(heading_error) >= config.CORNER_ENTER_HEADING
        normal_big_steer_now = steering is not None and abs(steering) >= config.CORNER_ENTER_STEER
        sharp_now = normal_sharp_now or normal_big_steer_now
        self.sharp_count = self.sharp_count + 1 if sharp_now else 0
        sharp_debounced = self.sharp_count >= config.CORNER_ENTER_DEBOUNCE_FRAMES

        weak_white_now = det["mode"] in ("LEFT_ONLY", "RIGHT_ONLY", "LOST")
        self.weak_white_count = self.weak_white_count + 1 if weak_white_now else 0
        weak_white = self.weak_white_count >= config.CORNER_ENTER_WEAK_WHITE_FRAMES

        if (not self.corner_active and corner_steering is not None and len(corner_path) >= 2 and
                (weak_white or sharp_debounced)):
            self.corner_active = True
            self.corner_exit_count = 0
            self.corner_hold_count = 0

        # 코너 이탈 판정
        if self.corner_active:
            center_recovered = (det["mode"] == "BOTH" and heading_error is not None and
                                 abs(heading_error) < config.CORNER_EXIT_HEADING and
                                 steering is not None and abs(steering) < config.CORNER_EXIT_STEER)
            self.corner_exit_count = self.corner_exit_count + 1 if center_recovered else 0
            if self.corner_exit_count >= config.CORNER_EXIT_BOTH_FRAMES:
                self.corner_active = False
                self.corner_exit_count = self.corner_hold_count = 0

        # 최종 조향 선택
        control_steering, control_mode = steering, "NORMAL"
        if self.corner_active:
            control_mode = "CORNER"
            if corner_steering is not None:
                control_steering = corner_steering
                self.last_corner_steer = corner_steering
                self.corner_hold_count = 0
            elif self.corner_hold_count < config.CORNER_HOLD_FRAMES:
                control_mode = "CORNER_HOLD"
                control_steering = self.last_corner_steer
                self.corner_hold_count += 1
            else:
                control_mode = "LOST"
                control_steering = None

        if control_steering is None:
            return self._lost_step(f"mode={det['mode']} no path")

        if abs(control_steering) > 1.0:
            self.last_steer_sign = 1.0 if control_steering > 0 else -1.0

        self.smoothed_steer = float(np.clip(
            config.STEER_ALPHA * control_steering + (1.0 - config.STEER_ALPHA) * self.smoothed_steer,
            -config.STEER_MAX, config.STEER_MAX))
        steering_for_speed = control_steering

        # 속도 계획
        if control_mode == "CORNER":
            corner_severity = min(1.0, abs(control_steering) / config.STEER_MAX)
            target_speed = config.SPEED_CORNER - (config.SPEED_CORNER - config.SPEED_CORNER_MIN) * corner_severity
        elif control_mode == "CORNER_HOLD":
            target_speed = config.SPEED_CORNER_HOLD
        else:
            target_speed = choose_target_speed(det, steering_for_speed, heading_error, preview_curvature)

        self.smoothed_target_speed = target_speed if self.smoothed_target_speed is None else (
            config.TARGET_SPEED_ALPHA * target_speed + (1.0 - config.TARGET_SPEED_ALPHA) * self.smoothed_target_speed)
        self.current_speed = ramp_speed(self.current_speed, self.smoothed_target_speed)

        # 장애물 회피 방향 힌트 설정
        if abs(self.smoothed_lateral_error) > config.LANE_BIAS_DEADBAND:
            curve_direction = -1.0 if self.smoothed_lateral_error > 0 else 1.0
        elif abs(self.smoothed_steer) >= config.LINE_CURVE_THRESHOLD:
            curve_direction = 1.0 if self.smoothed_steer > 0 else -1.0
        else:
            curve_direction = None

        self.last_debug = {"src_pts": src_pts, "bev": bev, "det": det,
                            "white_mask": white_mask, "yellow_mask": yellow_mask,
                            "corner_path": corner_path, "corner_target": corner_target}
        target_lane = getattr(config, "target_lane", "LEFT")
        status = (f"ctrl={control_mode} mode={det['mode']} conf={det['confidence']:.2f} "
                  f"L={det['left_count']} R={det['right_count']} lane={target_lane}")
        return self.smoothed_steer, self.current_speed, curve_direction, status

    def _lost_step(self, reason):
        search_steer = config.SEARCH_STEER if self.last_steer_sign > 0 else -config.SEARCH_STEER
        return search_steer, config.SEARCH_SPEED, None, f"lost -- searching ({reason})"
