"""차선 인식(순수 함수 위주) -- BEV 변환, 흰색 양쪽 경계선 슬라이딩 윈도우 추적(LaneTracker),
노란 점선 중앙선(코너 폴백) 검출. 상태는 LaneTracker.lane_width_px(최근 관측 차선폭 EMA) 하나뿐이고,
그 외에는 프레임 하나 받아서 결과 dict/리스트를 돌려주는 함수들입니다.

코너모드 진입/이탈 판단, 속도계획, Stanley 조향 EMA 같은 "상태 있는 제어 로직"은 여기 두지 않고
lane_keeper.py의 LaneKeeper가 이 모듈의 함수들을 호출해서 조합합니다.
"""
import math

import cv2
import numpy as np

import config


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
    """흰색 양쪽 경계선을 슬라이딩 윈도우로 x=f(y) 추적.
    양쪽 다 잡히면(BOTH) 그 중심을 쓰고, 한쪽만 잡히면(LEFT_ONLY/RIGHT_ONLY) 최근 관측한
    차선폭(lane_width_px)으로 중심을 추정. 둘 다 놓치면 LOST."""

    def __init__(self):
        self.lane_width_px = None

    def detect(self, mask):
        h, w = mask.shape
        y_near, y_far = int(h * config.NEAR_Y_RATIO), int(h * config.FAR_Y_RATIO)

        left_fit, left_count = sliding_window_fit(mask, "left")
        right_fit, right_count = sliding_window_fit(mask, "right")

        left_near, right_near = fit_x(left_fit, y_near), fit_x(right_fit, y_near)
        left_far, right_far = fit_x(left_fit, y_far), fit_x(right_fit, y_far)

        both = all(v is not None for v in [left_near, right_near, left_far, right_far])
        if both:
            min_w, max_w = w * config.LANE_WIDTH_MIN_RATIO, w * config.LANE_WIDTH_MAX_RATIO
            if not (min_w <= right_near - left_near <= max_w and
                    min_w <= right_far - left_far <= max_w and
                    left_near < right_near and left_far < right_far):
                both = False

        if both:
            observed = 0.5 * ((right_near - left_near) + (right_far - left_far))
            self.lane_width_px = (config.LANE_WIDTH_ALPHA * observed +
                                   (1 - config.LANE_WIDTH_ALPHA) * self.lane_width_px
                                   ) if self.lane_width_px else observed
            center_near, center_far = 0.5 * (left_near + right_near), 0.5 * (left_far + right_far)
            mode, confidence = "BOTH", 1.0
        elif self.lane_width_px is not None:
            if left_near is not None and left_far is not None:
                center_near = left_near + self.lane_width_px / 2.0
                center_far = left_far + self.lane_width_px / 2.0
                mode, confidence = "LEFT_ONLY", 0.60
            elif right_near is not None and right_far is not None:
                center_near = right_near - self.lane_width_px / 2.0
                center_far = right_far - self.lane_width_px / 2.0
                mode, confidence = "RIGHT_ONLY", 0.60
            else:
                center_near = center_far = None
                mode, confidence = "LOST", 0.0
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
    left, right = fit_x(det["left_fit"], y), fit_x(det["right_fit"], y)
    if left is not None and right is not None:
        return 0.5 * (left + right)
    width = det.get("lane_width_px")
    if width:
        if left is not None:
            return left + width / 2.0
        if right is not None:
            return right - width / 2.0
    return None


# ============================================================
# Stanley 조향
# ============================================================

def steering_from_lane(det, w, current_speed):
    """Stanley Controller 조향 계산. 횡오차 항을
    atan(K_LATERAL*0.1*lateral_error/(speed+STANLEY_SOFT_FACTOR))로 계산해서, 속도가
    낮을 때(코너)는 보정을 강하게, 속도가 높을 때(직선)는 자동으로 약하게(오실레이션 억제)."""
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
    """근접->원거리로 차선 중심을 샘플링해 미래 헤딩/곡률을 추정 (속도 계획에만 사용).
    주의: 이 값을 조향에 블렌딩하려 하지 마세요 -- 예전 버전들에서 그런 코드가 있었지만
    실제로는 다른 변수에 덮어써져 아무 효과가 없는 죽은 코드였습니다(리뷰에서 확인됨)."""
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
    """노란 점선 점들을 가까운 순서로 이어붙여 2D 경로를 만듦 (x=f(y) 방식과 달리 옆으로도 꺾일 수 있음)."""
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
# 학습된 CNN 폴백 (하이브리드 -- USE_LEARNED_LANE_MODEL)
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
