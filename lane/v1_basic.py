"""라인트레이싱 v1_basic -- 차선 인식 + LaneKeeper 컨트롤러를 한 파일로 통합.

차선 인식(순수 함수 위주): BEV 변환, 흰색 양쪽 경계선 슬라이딩 윈도우 추적(LaneTracker),
노란 점선 중앙선(코너 폴백) 검출, Stanley 조향 계산, 속도 계획. 상태는 LaneTracker의
lane_width_px(최근 관측 차선폭 EMA) 정도이고, 그 외엔 프레임 하나 받아서 값을 계산만 하는
함수들입니다.

LaneKeeper: 위 함수들의 결과를 받아 코너모드 진입/이탈 상태머신, 조향 EMA, 속도계획,
LOST 복구를 조합하는 상태 있는 컨트롤러. main()에서 한 번만 만들어서 루프 내내 재사용하고,
매 프레임 keeper.step(img) -> (line_steer, speed, curve_direction, status)를 호출합니다.
장애물 회피(obstacle/)는 여기서 하지 않고 main.py에서 line_steer에 더합니다.

[FIX] 원본(auto4)의 코너진입 디바운스(CORNER_ENTER_WEAK_WHITE_FRAMES)는 weak_white(흰선
LEFT_ONLY/RIGHT_ONLY/LOST) 조건에만 걸려있고, normal_sharp/normal_big_steer(조향각/헤딩
기반)는 단일 프레임만으로도 즉시 코너모드를 켤 수 있었습니다. 콘이 흰선을 살짝 왜곡시켜
폴리핏이 한 프레임 튀면 이 경로로 여전히 오진입할 수 있어서(코드 리뷰에서 확인), 이 두
조건에도 CORNER_ENTER_DEBOUNCE_FRAMES 프레임 연속 디바운스를 추가했습니다.
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


# ============================================================
# LaneKeeper -- 상태(코너모드/조향 EMA/속도 램프)를 들고 있는 컨트롤러
# ============================================================

class LaneKeeper:
    """흰색 양쪽 경계선(평상시, Stanley 조향) / 노란점선경로(급커브) 하이브리드 차선 추종.
    상태를 인스턴스에 들고 있으므로 main()에서 한 번만 만들어서 루프 내내 재사용합니다."""

    def __init__(self):
        self.tracker = LaneTracker()
        self.corner_active = False
        self.corner_exit_count = 0
        self.corner_hold_count = 0
        self.weak_white_count = 0   # weak_white 코너진입 조건 디바운스용
        self.sharp_count = 0        # [FIX] normal_sharp/normal_big_steer 디바운스용 (신규)
        self.smoothed_lateral_error = 0.0   # 라바콘 회피 방향 힌트용
        self.smoothed_center_near = None    # 조향 계산용 차선중심 스무딩
        self.smoothed_center_far = None
        self.last_corner_steer = 0.0
        self.smoothed_steer = 0.0
        self.current_speed = 0.0
        self.smoothed_target_speed = None
        self.last_steer_sign = 1.0
        self.last_debug = {}   # 디버그 뷰에서 참고할 최근 프레임 정보

    def step(self, img):
        """카메라 프레임 하나로 (line_steer, speed, curve_direction, status) 계산."""
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
        white_mask[:int(h * 0.18), :] = 0   # 위쪽(하늘/배경) 오탐 방지 -- make_white_mask 호출부의 필수 후처리

        det = self.tracker.detect(white_mask)

        # 조향 계산용 차선중심 스무딩 (CENTER_SMOOTH_ALPHA) -- 원본 det는 디버그 패널에 그대로
        # 쓰기 위해 별도 dict(det_for_steer)에만 스무딩된 값을 덮어씀
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

        # --- 코너 진입 판단 (디바운스 적용) ---
        normal_sharp_now = heading_error is not None and abs(heading_error) >= config.CORNER_ENTER_HEADING
        normal_big_steer_now = steering is not None and abs(steering) >= config.CORNER_ENTER_STEER
        # [FIX] 콘이 흰색 경계선을 잠깐 가려서(1프레임) 폴리핏이 튀는 것만으로 오진입하지 않도록,
        # weak_white와 동일하게 이 신호도 몇 프레임 연속돼야만 인정
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

        # --- 코너 이탈 판단 ---
        if self.corner_active:
            center_recovered = (det["mode"] == "BOTH" and heading_error is not None and
                                 abs(heading_error) < config.CORNER_EXIT_HEADING and
                                 steering is not None and abs(steering) < config.CORNER_EXIT_STEER)
            self.corner_exit_count = self.corner_exit_count + 1 if center_recovered else 0
            if self.corner_exit_count >= config.CORNER_EXIT_BOTH_FRAMES:
                self.corner_active = False
                self.corner_exit_count = self.corner_hold_count = 0

        # --- 최종 조향 선택 ---
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

        # --- 속도 계획 ---
        if control_mode == "CORNER":
            # 코너 조향각 크기(=코너가 얼마나 급한지)로 속도를 선형 보간 -- 고정값이면 코너
            # 진입 즉시 속도가 딱 튀는 느낌이 들어서, 급할수록 더 느리게 자동 조정
            corner_severity = min(1.0, abs(control_steering) / config.STEER_MAX)
            target_speed = config.SPEED_CORNER - (config.SPEED_CORNER - config.SPEED_CORNER_MIN) * corner_severity
        elif control_mode == "CORNER_HOLD":
            target_speed = config.SPEED_CORNER_HOLD
        else:
            target_speed = choose_target_speed(det, steering_for_speed, heading_error, preview_curvature)

        self.smoothed_target_speed = target_speed if self.smoothed_target_speed is None else (
            config.TARGET_SPEED_ALPHA * target_speed + (1.0 - config.TARGET_SPEED_ALPHA) * self.smoothed_target_speed)
        self.current_speed = ramp_speed(self.current_speed, self.smoothed_target_speed)

        # --- 장애물 회피 방향 힌트 ---
        # 코너 여부와 무관하게 lateral_error 치우침을 우선 사용하고, 애매하게 중심에 있을
        # 때만 조향 방향으로 보완 (직선 구간도 커버하기 위함). 스무딩된 값을 씀 -- 원시값은
        # 차가 중심 근처에서 살짝 흔들리기만 해도 프레임마다 부호가 뒤집혀서 회피 방향도
        # 매번 반대로 튀는 문제가 있었음.
        if abs(self.smoothed_lateral_error) > config.LANE_BIAS_DEADBAND:
            curve_direction = -1.0 if self.smoothed_lateral_error > 0 else 1.0
        elif abs(self.smoothed_steer) >= config.LINE_CURVE_THRESHOLD:
            curve_direction = 1.0 if self.smoothed_steer > 0 else -1.0
        else:
            curve_direction = None

        self.last_debug = {"src_pts": src_pts, "bev": bev, "det": det,
                            "white_mask": white_mask, "yellow_mask": yellow_mask,
                            "corner_path": corner_path, "corner_target": corner_target}
        status = (f"ctrl={control_mode} mode={det['mode']} conf={det['confidence']:.2f} "
                  f"L={det['left_count']} R={det['right_count']}")
        return self.smoothed_steer, self.current_speed, curve_direction, status

    def _lost_step(self, reason):
        """흰색 경계선을 완전히 놓쳤을 때: 마지막 방향으로 천천히 크리핑하며 즉시 재탐색.
        (그냥 정지하면 대회 중 사람 개입이 불가능해 영구 정지 위험 -- Stateless 요건상
        정지 단계 없이 바로 재탐색)"""
        search_steer = config.SEARCH_STEER if self.last_steer_sign > 0 else -config.SEARCH_STEER
        return search_steer, config.SEARCH_SPEED, None, f"lost -- searching ({reason})"
