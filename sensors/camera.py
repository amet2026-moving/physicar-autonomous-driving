# 카메라 파이프라인: BEV 원근변환(A) + 흰선 인식(B) + 노란선 인식(C) + 신호등 인식(D).
#
# 이 파일은 "무엇이 보이는가"만 답한다. 판단(코너인지/신호가 초록인지 등)은 절대
# 하지 않음 -- 예를 들어 노란선은 코너 여부와 상관없이 항상 인식해서 결과에 채워
# 넣고, 그걸 코너 신호로 볼지는 decision/lane_judge.is_corner()가 결정한다.
#
# 원본: T_T.py(흰선/노란선 로직), light_1.py(신호등 로직)의 실차 검증된 코드를
# 그대로 포팅. 튜닝 상수는 config/camera_params.py에 있음 -- 카메라 거치 각도가
# 바뀌면 이 파일이 아니라 그쪽을 고칠 것.
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import camera_params as cfg


# ============================================================
# 관측 결과 자료구조
# ============================================================

@dataclass
class LaneObservation:
    """한 프레임의 차선 인식 결과 (흰선 B + 노란선 C 통합)."""
    mode: str                                          # "BOTH"/"LEFT_ONLY"/"YELLOW_ONLY"/"LOST"
    confidence: float = 0.0                            # 0~1, mode별 신뢰도 (BOTH=1.0, *_ONLY=0.6, LOST=0.0)
    left_fit: np.ndarray | None = None                 # 왼쪽 흰선 다항식 계수 (x=ay^2+by+c), 없으면 None
    right_fit: np.ndarray | None = None                # 오른쪽 흰선 다항식 계수, 없으면 None
    left_count: int = 0                                # 왼쪽 피팅에 쓰인 픽셀 수
    right_count: int = 0                                # 오른쪽 피팅에 쓰인 픽셀 수
    center_near: float | None = None                   # 가까운 지점(y_near) 차선중심 x좌표 (BEV px)
    center_far: float | None = None                    # 먼 지점(y_far) 차선중심 x좌표 (BEV px)
    lane_width_px: float | None = None                 # 추정 차선폭 EMA (BEV px)
    y_near: int = 0                                     # center_near를 샘플링한 BEV y좌표 (px)
    y_far: int = 0                                       # center_far를 샘플링한 BEV y좌표 (px)
    yellow_points: list = field(default_factory=list)  # 노란 블롭 중심점 [(cx,cy,area), ...] (BEV px)
    yellow_path: list = field(default_factory=list)    # 노란점을 연결한 경로 [(x,y), ...] (BEV px)
    # ^ yellow_path는 코너/직선 구분 없이 노란선이 보이면 항상 채워짐. "이걸 코너로
    #   볼지"는 여기서 정하지 않는다 -- decision/lane_judge.is_corner()의 몫.
    bev_w: int = 0                                       # BEV 프레임 폭 (px) -- sensors/fusion.py가 좌표 정규화에 씀
    bev_h: int = 0                                       # BEV 프레임 높이 (px)


@dataclass
class TrafficLightObservation:
    """한 프레임의 신호등(RED-LOCK) 인식 결과. GREEN은 보지 않는다 -- decision/
    traffic_judge.py의 _RedLockGate가 RED의 등장/소실만으로 출발을 판단한다."""
    red_detected: bool = False    # 이번 프레임에 RED로 볼만한 컨투어가 있었는지
    red_ratio: float = 0.0        # 빨강 픽셀 비율 (검사 영역 대비, 0~1)
    red_pixels: int = 0           # 빨강 픽셀 총 개수 (개) -- 락 유지 판정의 두 번째 기준
    largest_area: float = 0.0     # 가장 큰 빨간 컨투어 면적 (px^2)
    valid_frame: bool = False     # 평균 명도가 너무 어둡지 않은 정상 프레임인지
    bbox: tuple | None = None     # 가장 큰 빨간 컨투어의 bbox (x1,y1,x2,y2, ROI-local px)


# draw_debug()가 마스크 원본을 다시 계산하지 않고 쓰도록 저장해두는 캐시.
# detect_lane_lines()/detect_red() 호출 시마다 최신으로 갱신됨.
_last_debug = {}


# ============================================================
# A. BEV_PERSPECTIVE_TRANSFORM -- 좌표계 변환
# ============================================================
# 카메라 원본 프레임의 사다리꼴 ROI를 위에서 내려다본(BEV) 직사각형으로 펴는
# 원근변환. B/C 섹션은 전부 이 BEV 좌표계 위에서 동작한다.

def roi_points_px(w, h):
    """ROI_NORM(정규화 0~1 좌표)을 실제 프레임 크기(w,h) 기준 픽셀 좌표로 변환."""
    pts = cfg.ROI_NORM.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.float32)


def build_bev_matrices(w, h):
    """ROI 사다리꼴 -> 직사각형 BEV로 바꾸는 원근변환 행렬 M과 역행렬 Minv 생성."""
    src = roi_points_px(w, h)
    margin = 0.15 * w   # 사다리꼴 윗변보다 넓게 잡아 BEV 좌우가 잘리지 않게 함

    dst = np.float32([
        [margin, 0],
        [w - margin, 0],
        [w - margin, h - 1],
        [margin, h - 1],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return src, M, Minv


def warp_to_bev(frame):
    """원본 프레임을 BEV로 원근변환. (bev_frame, M) 반환."""
    h, w = frame.shape[:2]
    _src_pts, M, _Minv = build_bev_matrices(w, h)
    bev = cv2.warpPerspective(frame, M, (w, h))
    return bev, M


# ============================================================
# B. WHITE_LINE_PERCEPTION -- 흰선 인식 (마스크 생성 -> 다항식 피팅)
# ============================================================
# B-1: 밝기 변화에 강건한 흰색 픽셀 분류(segmentation).
# B-2: 그 마스크에 슬라이딩 윈도우 + 다항식 회귀로 좌/우 경계선을 피팅하고,
#      차선폭을 프레임 간 EMA로 추적한다 (LaneTracker, 상태 보유).

def _adaptive_asphalt_context(bev_bgr):
    """차량 바로 앞 코리도를 샘플링해 '지금 이 프레임의 도로 밝기'를 추정하고,
    그 기준으로 도로(아스팔트) 마스크를 만든다. 고정 임계값 대신 이걸 쓰는 이유:
    노출/밝기가 바뀌면 같은 아스팔트도 밝기값이 흔들리기 때문. 흰선(B-2)과
    노란선(C-1) 마스크가 공유해서 쓰는 헬퍼."""
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h, w = v.shape

    y1, y2 = int(h * 0.62), max(int(h * 0.96), int(h * 0.62) + 1)
    x1, x2 = int(w * 0.28), max(int(w * 0.72), int(w * 0.28) + 1)
    seed_v = v[y1:y2, x1:x2]
    seed_s = s[y1:y2, x1:x2]

    usable = seed_v[seed_s <= cfg.ASPHALT_S_MAX]
    if usable.size < 40:
        usable = seed_v.reshape(-1)

    # 60분위수를 쓰는 이유: 중앙값보다 살짝 위로 잡아야 약간 밝은 도로도 하나로
    # 이어지고, 그렇다고 흰선/노란선 같은 희소하고 밝은 픽셀에 끌려가지도 않는다.
    road_v_ref = float(np.percentile(usable, 60)) if usable.size else 120.0
    road_v_max = float(np.clip(
        road_v_ref + cfg.ASPHALT_ADAPTIVE_V_MARGIN,
        cfg.ASPHALT_ADAPTIVE_V_MIN,
        cfg.ASPHALT_ADAPTIVE_V_MAX,
    ))

    road = ((v <= road_v_max) & (s <= cfg.ASPHALT_S_MAX)).astype(np.uint8) * 255
    return road, road_v_ref, road_v_max


def make_white_mask(bev_bgr):
    """밝기 변화에 강건한 흰색 차선 마스크. 밝은 잔디가 흰선처럼 보이는 걸
    막기 위해 (밝기 + 무채색 + 국소대비 + 도로인접) 네 조건을 모두 요구한다."""
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2LAB)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    road, road_v_ref, _ = _adaptive_asphalt_context(bev_bgr)
    white_v_min = float(np.clip(
        road_v_ref + cfg.WHITE_ADAPTIVE_V_MARGIN,
        cfg.WHITE_V_MIN,
        cfg.WHITE_ADAPTIVE_V_MAX,
    ))

    # 흰/회색 페인트는 R/G/B가 서로 비슷(무채색)하지만 잔디는 밝아져도 여전히
    # 유채색이다 -- Lab a/b채널로 조명 변화에 덜 민감한 2차 검증을 더한다.
    b, g, r = cv2.split(bev_bgr)
    max_c = np.maximum(np.maximum(b, g), r).astype(np.int16)
    min_c = np.minimum(np.minimum(b, g), r).astype(np.int16)
    rgb_spread = max_c - min_c

    a = lab[:, :, 1].astype(np.int16)
    bb = lab[:, :, 2].astype(np.int16)
    neutral = (
        (rgb_spread <= cfg.WHITE_RGB_SPREAD_MAX)
        & (np.abs(a - 128) <= cfg.WHITE_LAB_A_TOL)
        & (np.abs(bb - 128) <= cfg.WHITE_LAB_B_TOL)
    )

    bright_neutral = (v >= white_v_min) & (s <= cfg.WHITE_S_MAX) & neutral

    # 탑햇 필터: 흰선은 어두운 아스팔트 위의 좁고 밝은 구조물이므로, 잔디처럼
    # 넓게 밝은 영역은 이걸로 걸러진다.
    tk = max(5, int(cfg.WHITE_TOPHAT_KERNEL))
    if tk % 2 == 0:
        tk += 1
    top_hat = cv2.morphologyEx(
        v, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tk, tk)),
    )
    local_line = top_hat >= cfg.WHITE_LOCAL_CONTRAST_MIN

    # 탑햇 커널보다 굵은, 확실히 밝은 흰선을 위한 예외 통로 (무채색/도로인접
    # 조건은 그대로 적용되므로 잔디까지 통과하지는 않음).
    very_bright = v >= min(252.0, white_v_min + 24.0)

    dk = max(3, int(cfg.ROAD_CONTEXT_DILATE))
    if dk % 2 == 0:
        dk += 1
    road_near = cv2.dilate(
        road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)), iterations=1,
    ) > 0

    mask_bool = bright_neutral & road_near & (local_line | very_bright)
    mask = mask_bool.astype(np.uint8) * 255

    kernel = np.ones((cfg.MORPH_KERNEL, cfg.MORPH_KERNEL), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _search_base(histogram, x0, x1):
    """히스토그램의 [x0,x1) 구간에서 최댓값 위치(경계선 시작 x좌표)를 찾는다."""
    x0 = max(0, int(x0))
    x1 = min(len(histogram), int(x1))
    if x1 <= x0:
        return None
    section = histogram[x0:x1]
    if section.size == 0 or np.max(section) <= 0:
        return None
    return int(x0 + np.argmax(section))


def sliding_window_fit(binary, side):
    """마스크에서 한쪽(left/right) 경계선을 슬라이딩 윈도우로 따라가며
    x = f(y) 2차 다항식으로 피팅. 반환: (다항식 계수 또는 None, 사용된 픽셀 수)."""
    h, w = binary.shape
    histogram = np.sum(binary[int(h * 0.45):, :] > 0, axis=0)

    a, b = cfg.LEFT_SEARCH if side == "left" else cfg.RIGHT_SEARCH
    base = _search_base(histogram, a * w, b * w)
    if base is None:
        return None, 0

    nonzero_y, nonzero_x = binary.nonzero()
    window_h = max(1, h // cfg.NWINDOWS)
    margin = max(10, int(w * cfg.WINDOW_MARGIN_RATIO))
    x_current = base
    lane_inds = []

    for win in range(cfg.NWINDOWS):
        y_low = h - (win + 1) * window_h
        y_high = h - win * window_h
        x_low = x_current - margin
        x_high = x_current + margin

        good = (
            (nonzero_y >= y_low) & (nonzero_y < y_high)
            & (nonzero_x >= x_low) & (nonzero_x < x_high)
        )
        inds = np.where(good)[0]
        if inds.size:
            lane_inds.append(inds)
        if inds.size >= cfg.MINPIX:
            x_current = int(np.mean(nonzero_x[inds]))   # 윈도우 중심을 실제 픽셀 쪽으로 재조정

    if not lane_inds:
        return None, 0

    lane_inds = np.concatenate(lane_inds)
    xs = nonzero_x[lane_inds]
    ys = nonzero_y[lane_inds]

    if len(xs) < cfg.MIN_FIT_PIXELS:
        return None, len(xs)   # 픽셀이 너무 적으면 피팅하지 않고 개수만 보고

    fit = np.polyfit(ys, xs, 2)
    return fit, len(xs)


def fit_x(fit, y):
    """다항식 계수로 특정 y에서의 x값 계산. fit이 None이면 None 반환."""
    if fit is None:
        return None
    return float(fit[0] * y * y + fit[1] * y + fit[2])


class LaneTracker:
    """왼쪽 흰선 + 노란 점선 사이 코리도를 기본 주행선으로 추적한다(팀 결정: 왼쪽
    차로 주행, 장애물 시 오른쪽으로 회피). 오른쪽 흰선(right_fit)은 디버그 시각화용
    으로만 계속 계산하고, 코리도 중심 계산에는 쓰지 않는다 -- 그건 도로 반대쪽 바깥
    경계일 뿐, 지금 달리는 코리도의 경계가 아니기 때문.
    한쪽만 보여도 지난 코리도 폭(EMA) 기억으로 반대쪽 경계를 추정한다
    (LEFT_ONLY/YELLOW_ONLY). 노란선 x좌표는 C 섹션이 만든 yellow_path를
    _yellow_x_at_row()로 y_near/y_far에서 보간해서 얻는다 -- 단, 점이
    YELLOW_CORRIDOR_MIN_POINTS 미만이면(점 1개는 near/far가 항상 같은 값으로
    보간돼 노이즈에 취약) 아예 안 본 것으로 치고 LEFT_ONLY 폴백으로 넘어간다."""

    def __init__(self):
        self.lane_width_px = None   # 코리도 폭 EMA (BEV px, 왼쪽흰선~노란선), 처음엔 미지

    def detect(self, mask, yellow_path=None) -> LaneObservation:
        h, w = mask.shape
        y_near = int(h * cfg.NEAR_Y_RATIO)
        y_far = int(h * cfg.FAR_Y_RATIO)

        left_fit, left_count = sliding_window_fit(mask, "left")
        right_fit, right_count = sliding_window_fit(mask, "right")   # 디버그 시각화 전용

        left_near, left_far = fit_x(left_fit, y_near), fit_x(left_fit, y_far)

        # 점이 YELLOW_CORRIDOR_MIN_POINTS 미만이면 아예 안 본 것으로 취급한다.
        # 점 1개짜리는 near/far 보간이 항상 같은 값(np.interp가 점 하나면 상수를
        # 반환)이라, 코너 진입 직전처럼 노란 점이 막 나타나기 시작하는 노이즈에
        # 민감한 순간에 center_near/far를 엉뚱한 방향으로 끌고 갈 수 있다.
        has_reliable_yellow = len(yellow_path or []) >= cfg.YELLOW_CORRIDOR_MIN_POINTS
        yellow_near = _yellow_x_at_row(yellow_path, y_near) if has_reliable_yellow else None
        yellow_far = _yellow_x_at_row(yellow_path, y_far) if has_reliable_yellow else None

        both = (
            left_near is not None and yellow_near is not None
            and left_far is not None and yellow_far is not None
        )

        if both:
            width_near = yellow_near - left_near
            width_far = yellow_far - left_far
            min_w, max_w = w * cfg.LANE_WIDTH_MIN_RATIO, w * cfg.LANE_WIDTH_MAX_RATIO
            geometry_ok = (
                min_w <= width_near <= max_w and min_w <= width_far <= max_w
                and left_near < yellow_near and left_far < yellow_far
            )
            if not geometry_ok:
                both = False   # 순서가 뒤바뀌었거나 폭이 비정상이면 둘 다 못 찾은 것으로 취급

        if both:
            observed_width = 0.5 * ((yellow_near - left_near) + (yellow_far - left_far))
            self.lane_width_px = (
                observed_width if self.lane_width_px is None
                else cfg.LANE_WIDTH_ALPHA * observed_width + (1.0 - cfg.LANE_WIDTH_ALPHA) * self.lane_width_px
            )
            center_near = 0.5 * (left_near + yellow_near)
            center_far = 0.5 * (left_far + yellow_far)
            mode, confidence = "BOTH", 1.0

        elif self.lane_width_px is not None and left_near is not None and left_far is not None:
            center_near = left_near + self.lane_width_px / 2.0
            center_far = left_far + self.lane_width_px / 2.0
            mode, confidence = "LEFT_ONLY", 0.60

        elif self.lane_width_px is not None and yellow_near is not None and yellow_far is not None:
            center_near = yellow_near - self.lane_width_px / 2.0
            center_far = yellow_far - self.lane_width_px / 2.0
            mode, confidence = "YELLOW_ONLY", 0.60

        else:
            center_near = center_far = None
            mode, confidence = "LOST", 0.0

        if center_near is not None and not (-0.15 * w <= center_near <= 1.15 * w):
            center_near = center_far = None   # 화면 밖으로 튄 추정치는 버림
            mode, confidence = "LOST", 0.0

        return LaneObservation(
            mode=mode, confidence=confidence,
            left_fit=left_fit, right_fit=right_fit,
            left_count=left_count, right_count=right_count,
            center_near=center_near, center_far=center_far,
            lane_width_px=self.lane_width_px,
            y_near=y_near, y_far=y_far,
        )


_lane_tracker = LaneTracker()   # 모듈 전역 싱글턴 -- detect_lane_lines()가 매 프레임 재사용


# ============================================================
# C. YELLOW_LINE_PERCEPTION -- 노란선 인식 (마스크 -> centroid -> 경로 연결)
# ============================================================
# 코너인지 아닌지는 여기서 신경 쓰지 않는다. 노란선이 보이면 조건 없이 항상
# 마스크/점/경로를 뽑는다. "이걸 코너 신호로 볼지"는 decision/lane_judge.is_corner()가
# 결정한다.

def make_yellow_mask(bev_bgr):
    """노란/주황 점선 마스크. 아스팔트 위/근처에 있는 조각만 남긴다(다른 노란
    물체 오탐 방지)."""
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array([cfg.YELLOW_H_MIN, cfg.YELLOW_S_MIN, cfg.YELLOW_V_MIN], dtype=np.uint8),
        np.array([cfg.YELLOW_H_MAX, 255, 255], dtype=np.uint8),
    )

    asphalt, _, _ = _adaptive_asphalt_context(bev_bgr)
    k = np.ones((cfg.ASPHALT_DILATE, cfg.ASPHALT_DILATE), np.uint8)
    asphalt_near = cv2.dilate(asphalt, k, iterations=1)
    yellow = cv2.bitwise_and(yellow, asphalt_near)

    k3 = np.ones((3, 3), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k3)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k3)
    return yellow


def yellow_centroids(yellow_mask):
    """노란 마스크의 각 블롭(점선 조각) 중심좌표와 면적을 추출: [(cx,cy,area), ...]."""
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = yellow_mask.shape
    pts = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.YELLOW_MIN_AREA or area > cfg.YELLOW_MAX_AREA:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
        if not (0.03 * w <= cx <= 0.97 * w) or not (0.05 * h <= cy <= 0.98 * h):
            continue   # 화면 가장자리 잡음 제외
        pts.append((cx, cy, area))

    return pts


def build_yellow_path(points, w, h):
    """노란 블롭 중심점들을 차량에서 가까운 순으로 체인처럼 연결해 2D 경로를
    만든다. x=f(y) 피팅과 달리 급격히 옆으로 꺾이는 형태도 표현할 수 있다."""
    if not points:
        return []

    start = np.array([w / 2.0, h - 8.0], dtype=np.float32)   # 차량 위치 근사(BEV 하단 중앙)
    unused = list(points)

    dists = [np.linalg.norm(np.array([p[0], p[1]], dtype=np.float32) - start) for p in unused]
    first_i = int(np.argmin(dists))
    if dists[first_i] > cfg.YELLOW_PATH_FIRST_LINK_PX:
        return []   # 가장 가까운 점조차 너무 멀면 경로 없음으로 취급

    first = unused.pop(first_i)
    path = [(first[0], first[1])]
    current = np.array([first[0], first[1]], dtype=np.float32)

    for _ in range(14):   # 점선 조각 최대 14개까지만 연결 (그 이상은 다른 물체일 가능성)
        if not unused:
            break

        best_i, best_cost = None, None
        for i, p in enumerate(unused):
            q = np.array([p[0], p[1]], dtype=np.float32)
            dist = float(np.linalg.norm(q - current))
            if dist > cfg.YELLOW_PATH_MAX_LINK_PX:
                continue

            # 화면상 "전방"은 y가 줄어드는 방향. 급코너에서는 y가 비슷한 채 x만
            # 바뀔 수 있어 약간의 역방향은 허용하되 페널티를 준다.
            backward = max(0.0, float(p[1] - current[1] - cfg.YELLOW_PATH_BACKWARD_ALLOW_PX))
            cost = dist + 3.0 * backward
            if best_cost is None or cost < best_cost:
                best_cost, best_i = cost, i

        if best_i is None:
            break
        p = unused.pop(best_i)
        path.append((p[0], p[1]))
        current = np.array([p[0], p[1]], dtype=np.float32)

    return path


def _yellow_x_at_row(yellow_path, target_y):
    """노란 경로 점들에서 target_y(BEV row)의 x값을 선형보간. 점이 없으면 None,
    범위를 벗어나면 가장 가까운 끝점 값으로 고정(clamp)된다. LaneTracker가 코리도
    오른쪽 경계(노란선) 위치를 y_near/y_far에서 구하는 데 쓴다."""
    if not yellow_path:
        return None
    pts = sorted(yellow_path, key=lambda p: -p[1])   # y 내림차순(가까운 점 먼저)
    neg_ys = [-p[1] for p in pts]                     # np.interp는 x가 증가해야 하므로 부호 반전
    xs = [p[0] for p in pts]
    return float(np.interp(-target_y, neg_ys, xs))


def detect_yellow_path(bev_frame):
    """C 섹션 공개 진입점: BEV 이미지 -> (노란 마스크, 중심점 리스트, 연결된 경로)."""
    mask = make_yellow_mask(bev_frame)
    points = yellow_centroids(mask)
    h, w = mask.shape
    path = build_yellow_path(points, w, h)
    return mask, points, path


# ============================================================

# 흰선(B) + 노란선(C) 통합 진입점

# ============================================================

def detect_lane_lines(bev_frame) -> LaneObservation:
    """B(흰선) + C(노란선) 결과를 하나의 LaneObservation으로 합쳐 반환.
    decision/lane_judge.py는 이 하나만 받아서 STRAIGHT/CORNER/OFF_TRACK을 판단한다.
    노란선을 코리도 오른쪽 경계로 쓰는 LaneTracker.detect()가 yellow_path를 필요로
    하므로, 노란선 인식(C)을 흰선 인식(B)보다 먼저 끝내둔다."""
    white_mask = make_white_mask(bev_frame)
    yellow_mask, yellow_points, yellow_path = detect_yellow_path(bev_frame)

    obs = _lane_tracker.detect(white_mask, yellow_path)
    obs.yellow_points = yellow_points
    obs.yellow_path = yellow_path
    obs.bev_h, obs.bev_w = bev_frame.shape[:2]

    _last_debug["white_mask"] = white_mask
    _last_debug["yellow_mask"] = yellow_mask
    return obs


# ============================================================
# D. TRAFFIC_LIGHT_PERCEPTION -- 신호등 RED 인식 (TTTTTT_physicar_ros2_red_lock_myapp.py 이식)
# ============================================================
# GREEN은 보지 않는다. ROI(또는 judge 쪽이 넘겨주는 락 패치, focus_box) 안에서 순수
# HSV로 빨강만 찾는다 -- "락 성립 후 그 자리만 계속 보기"는 decision/traffic_judge.py
# 의 _RedLockGate가 focus_box를 통해 담당하고(판단), 여기는 "그 영역 안에 지금 빨강이
# 얼마나 있는가"만 답한다(인식). 판단과 인식을 분리한 이 프로젝트 컨벤션에 맞춰,
# 원본 파일의 lock 상태 관리(TrafficLightStartDetector.lock_box 등)는 여기 두지 않음.

def traffic_crop_roi(frame):
    """고정 ROI(FIXED_ROI_NORM)만 잘라낸다. (roi, roi_box=(x1,y1,x2,y2) 프레임 px) 반환."""
    h, w = frame.shape[:2]
    x1n, y1n, x2n, y2n = cfg.FIXED_ROI_NORM
    x1 = int(np.clip(x1n * w, 0, w - 1))
    y1 = int(np.clip(y1n * h, 0, h - 1))
    x2 = int(np.clip(x2n * w, x1 + 1, w))
    y2 = int(np.clip(y2n * h, y1 + 1, h))
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def make_red_mask(bgr):
    """HSV 두 구간(Hue 0 부근 + 180 부근, 빨강의 색상환 랩어라운드) OR로 빨강
    마스크를 만든다. valid_frame(평균 명도가 너무 어둡지 않은 정상 프레임인지)도
    같이 반환 -- 신호등 자체가 안 보이는 게 아니라 카메라가 통째로 깜깜한 경우를
    "빨강 없음"과 구분하기 위함."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    valid_frame = float(np.mean(hsv[:, :, 2])) >= cfg.TRAFFIC_MIN_ROI_MEAN_V

    low1 = np.array(cfg.TRAFFIC_RED_LOW_1, dtype=np.uint8)
    high1 = np.array(cfg.TRAFFIC_RED_HIGH_1, dtype=np.uint8)
    low2 = np.array(cfg.TRAFFIC_RED_LOW_2, dtype=np.uint8)
    high2 = np.array(cfg.TRAFFIC_RED_HIGH_2, dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, low1, high1), cv2.inRange(hsv, low2, high2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.TRAFFIC_MORPH_KERNEL_SIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, valid_frame


def _red_contour_stats(mask):
    """마스크에서 (빨강 픽셀 수, 빨강 비율, 최댓값 컨투어 면적, 최댓값 컨투어의
    bbox=(x1,y1,x2,y2)) 반환. 컨투어가 하나도 없으면 bbox는 None."""
    red_pixels = int(cv2.countNonZero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    red_ratio = red_pixels / float(total_pixels) if total_pixels > 0 else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0.0
    largest_bbox = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            largest_bbox = (x, y, x + w, y + h)

    return red_pixels, red_ratio, largest_area, largest_bbox


def detect_red(frame, focus_box=None) -> TrafficLightObservation:
    """D 섹션 공개 진입점: 원본 프레임 -> ROI(+선택적으로 그 안의 focus_box만) 안의
    RED 관측 결과. focus_box(ROI-local px, x1,y1,x2,y2)가 주어지면 그 패치 안만
    검사하되(락 후 다른 곳의 빨간 반사광/노이즈를 걸러내기 위함), 반환하는 bbox는
    항상 ROI-local 좌표로 환산해서 준다 -- 호출부가 락 여부와 무관하게 같은
    좌표계만 다루면 되게 하기 위함."""
    roi, _roi_box = traffic_crop_roi(frame)
    if roi.size == 0:
        return TrafficLightObservation()

    offset_x, offset_y = 0, 0
    search_area = roi

    if focus_box is not None:
        rh, rw = roi.shape[:2]
        fx1, fy1, fx2, fy2 = focus_box
        fx1 = max(0, min(int(fx1), rw - 1))
        fy1 = max(0, min(int(fy1), rh - 1))
        fx2 = max(fx1 + 1, min(int(fx2), rw))
        fy2 = max(fy1 + 1, min(int(fy2), rh))
        patch = roi[fy1:fy2, fx1:fx2]
        if patch.size == 0:
            return TrafficLightObservation()
        offset_x, offset_y = fx1, fy1
        search_area = patch

    mask, valid_frame = make_red_mask(search_area)
    red_pixels, red_ratio, largest_area, local_bbox = _red_contour_stats(mask)

    bbox = None
    if local_bbox is not None:
        lx1, ly1, lx2, ly2 = local_bbox
        bbox = (lx1 + offset_x, ly1 + offset_y, lx2 + offset_x, ly2 + offset_y)

    red_detected = (
        valid_frame
        and red_ratio >= cfg.TRAFFIC_RED_MIN_RATIO
        and largest_area >= cfg.TRAFFIC_RED_MIN_AREA
    )

    _last_debug["traffic_mask"] = mask
    _last_debug["traffic_focus_box"] = focus_box

    return TrafficLightObservation(
        red_detected=red_detected, red_ratio=red_ratio, red_pixels=red_pixels,
        largest_area=largest_area, valid_frame=valid_frame, bbox=bbox,
    )


# ============================================================
# DEBUG VISUALIZATION -- A~D 인식 결과를 한 이미지로 합성
# ============================================================

def _draw_polyline_fit(img, fit, color, thickness=3):
    """다항식 fit을 이미지 위에 곡선으로 그린다 (디버그 전용)."""
    if fit is None:
        return
    h, w = img.shape[:2]
    pts = []
    for y in np.linspace(0, h - 1, 80):
        x = fit_x(fit, y)
        if x is not None and -50 <= x <= w + 50:
            pts.append((int(x), int(y)))
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_red_bbox(img, bbox, color, label):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_debug(frame, bev_frame, lane_obs: LaneObservation, traffic_obs: TrafficLightObservation) -> np.ndarray:
    """A/B/C/D 인식 결과를 3행 2열 패널 하나로 합성.
    주의: 이번 프레임에 대해 detect_lane_lines()/detect_red()를 먼저 호출한 뒤에
    불러야 한다 (마스크 원본을 _last_debug 캐시에서 읽음)."""
    h, w = frame.shape[:2]

    # ---- 1행 좌: 원본 + ROI 사다리꼴 (A) ----
    src_pts = roi_points_px(w, h)
    cam_vis = frame.copy()
    cv2.polylines(cam_vis, [src_pts.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.putText(cam_vis, "A: CAMERA + ROI", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # ---- 1행 우: BEV + 흰선 피팅(B) + 노란 경로(C) ----
    bev_vis = bev_frame.copy()
    _draw_polyline_fit(bev_vis, lane_obs.left_fit, (0, 255, 0))
    _draw_polyline_fit(bev_vis, lane_obs.right_fit, (0, 255, 0))

    if lane_obs.center_near is not None and lane_obs.center_far is not None:
        cv2.line(bev_vis, (int(lane_obs.center_near), lane_obs.y_near),
                  (int(lane_obs.center_far), lane_obs.y_far), (0, 255, 255), 2)
        cv2.circle(bev_vis, (int(lane_obs.center_near), lane_obs.y_near), 6, (0, 255, 255), -1)

    for cx, cy, _area in lane_obs.yellow_points:
        cv2.circle(bev_vis, (int(cx), int(cy)), 4, (255, 0, 255), -1)
    if len(lane_obs.yellow_path) >= 2:
        pts = np.array([(int(x), int(y)) for x, y in lane_obs.yellow_path], dtype=np.int32)
        cv2.polylines(bev_vis, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(bev_vis, "B+C: BEV  green=white fit  cyan=center  magenta=yellow",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ---- 2행 좌: 흰색 마스크 (B) ----
    white_mask = _last_debug.get("white_mask")
    white_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR) if white_mask is not None else np.zeros_like(frame)
    cv2.putText(white_vis, "B: WHITE MASK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # ---- 2행 우: 노란 마스크 (C) ----
    yellow_mask = _last_debug.get("yellow_mask")
    yellow_vis = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR) if yellow_mask is not None else np.zeros_like(frame)
    cv2.putText(yellow_vis, "C: YELLOW MASK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # ---- 3행 좌: 신호등 ROI + 빨강 마스크 (D) ----
    traffic_mask = _last_debug.get("traffic_mask")
    if traffic_mask is not None:
        tl_vis = cv2.cvtColor(traffic_mask, cv2.COLOR_GRAY2BGR)
        _draw_red_bbox(tl_vis, traffic_obs.bbox, (0, 0, 255), "RED")
        tl_vis = cv2.resize(tl_vis, (w, h), interpolation=cv2.INTER_AREA)
    else:
        tl_vis = np.zeros_like(frame)
    cv2.putText(tl_vis, "D: TRAFFIC LIGHT RED MASK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---- 3행 우: 텍스트 상태 패널 ----
    info = np.zeros_like(frame)
    lane_width = lane_obs.lane_width_px
    lines = [
        f"[B] mode={lane_obs.mode}  confidence={lane_obs.confidence:.2f}",
        f"[B] lane_width_px={lane_width:.1f}" if lane_width is not None else "[B] lane_width_px=None",
        f"[C] yellow_points={len(lane_obs.yellow_points)}  path_len={len(lane_obs.yellow_path)}",
        f"[D] red_detected={traffic_obs.red_detected}  red_ratio={traffic_obs.red_ratio:.4f}",
        f"[D] largest_area={traffic_obs.largest_area:.1f}  valid_frame={traffic_obs.valid_frame}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(info, text, (18, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    top = np.hstack([cam_vis, bev_vis])
    mid = np.hstack([white_vis, yellow_vis])
    bottom = np.hstack([tl_vis, info])
    return np.vstack([top, mid, bottom])


# ============================================================
# 단독 실행: 인식 결과(A~D)만 웹 디버그 뷰로 확인
# ============================================================
# lidar/fusion/decision/control이 아직 없어 main.py 전체 파이프라인은 돌릴 수
# 없다. 이 블록은 camera.py 하나만 떼어내 "인식이 맞게 나오는가"부터 확인하는
# 용도 -- drive()는 아예 호출하지 않는다(차량 제어 없이 인식만).
if __name__ == "__main__":
    from config import base as base_cfg
    from hardware import car_api
    from myapp import debug_view

    debug_view.serve()
    print("[camera.py] perception-only self test -- http://localhost:5000")

    try:
        while True:
            t0 = time.time()
            frame = car_api.camera()
            if frame is None:
                time.sleep(0.2)
                continue

            bev, _M = warp_to_bev(frame)
            lane_obs = detect_lane_lines(bev)
            traffic_obs = detect_red(frame)
            panel = draw_debug(frame, bev, lane_obs, traffic_obs)

            status = (
                f"lane={lane_obs.mode} conf={lane_obs.confidence:.2f} "
                f"yellow_pts={len(lane_obs.yellow_points)} "
                f"tl_red={traffic_obs.red_detected} tl_ratio={traffic_obs.red_ratio:.4f}"
            )
            debug_view.update_web(panel, status)

            proc_time = time.time() - t0
            time.sleep(max(0.0, 1.0 / base_cfg.TARGET_FPS - proc_time))
    except KeyboardInterrupt:
        print("\n[camera.py] stopped")
    finally:
        debug_view.stop_view()
