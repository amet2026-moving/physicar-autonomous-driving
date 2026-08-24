#!/usr/bin/env python3

"""
평가결과: 
# 두번째 코너에서 이탈, 다중코너구간 장애물에서 무한멈춤
# 도착지 앞 마지막 장애물 앞에서 무한 멈춤, 약 80초
특이사항: 
# 코너나 라이다 통과 시 매우 느려짐

"""

import math
import socket
import threading
import time

import cv2
import numpy as np
import requests

# ============================================================
# LiDAR READ-ONLY CLUSTER LOGGER
# - /scan을 읽고 cluster 정보만 로그로 남깁니다.
# - speed / steering / lane / corner / traffic 제어에는 관여하지 않습니다.
# ============================================================
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

BASE_URL = "http://localhost"
session = requests.Session()

# ============================================================
# 0. TEST MODE
# ============================================================

# FIRST TEST MUST STAY FALSE.
# Change to True only after the debug view is correct.
DRIVE_ENABLED = True

TARGET_FPS = 15.0

# Speed
# Actual normal-driving speed is controlled by SPEED_MIN / SPEED_MAX
# and preview curvature below.
SPEED_ONE_LINE = 0.34

# Corner
SPEED_CORNER      = 0.28
SPEED_CORNER_HOLD = 0.24

# Acceleration / braking
SPEED_RAMP_UP   = 0.12
SPEED_RAMP_DOWN = 0.06

MAX_STEERING_DEG = 20.0

# Terminal status print interval
PRINT_INTERVAL = 1.0

# ============================================================
# 1. CAMERA ROI / BEV
# ============================================================

# Normalized ROI points: top-left, top-right, bottom-right, bottom-left
# Initial values only. Tune after looking at the actual camera image.
ROI_NORM = np.float32([
    [0.33, 0.57],   # top-left
    [0.67, 0.57],   # top-right
    [0.98, 0.92],   # bottom-right
    [0.02, 0.92],   # bottom-left
])

NEAR_Y_RATIO = 0.82
FAR_Y_RATIO  = 0.52

# ============================================================
# 2. WHITE MASK
# ============================================================

# White: low-ish saturation + high brightness
# Initial simulator values; tune if needed.
WHITE_S_MAX = 65
WHITE_V_MIN = 150

MORPH_KERNEL = 3

# ============================================================
# 3. BOUNDARY TRACKER
# ============================================================

NWINDOWS = 9
WINDOW_MARGIN_RATIO = 0.09
MINPIX = 18
MIN_FIT_PIXELS = 70

LEFT_SEARCH  = (0.03, 0.48)
RIGHT_SEARCH = (0.52, 0.97)

LANE_WIDTH_MIN_RATIO = 0.30
LANE_WIDTH_MAX_RATIO = 0.95
LANE_WIDTH_ALPHA = 0.20

# ============================================================
# 4. STEERING
# ============================================================

# Steering gains confirmed from the latest curve test direction.
K_LATERAL = 14.0
K_HEADING = 40.0
STEER_ALPHA = 0.55

# ------------------------------------------------------------
# Preview-based speed / steering
# ------------------------------------------------------------
# Preview geometry is used ONLY for speed planning.
# Fast -> sample slightly farther ahead.
PREVIEW_MIN_Y_RATIO = 0.44

# Speed planner limits.
SPEED_MIN = 0.28
SPEED_MAX = 0.75

# Do not slow down for tiny lane-fit / preview fluctuations.
# Only curvature above this level is considered a real bend.
CURVE_DEADBAND = 0.30

# Below this level, explicitly allow full straight-line speed.
STRAIGHT_CURVE_LIMIT = 0.18

# How quickly target speed itself is smoothed before the actuator ramp.
TARGET_SPEED_ALPHA = 0.45




# ============================================================
# 4-A. TRAFFIC LIGHT START GATE
# ============================================================
# Simulator baseline verified on RED/GREEN.
# IMPORTANT:
# - No assumption that RED is above/below GREEN.
# - Only a fixed broad signal ROI is used.
# - RED / UNKNOWN => WAIT.
# - GREEN must be seen for consecutive frames.
# - Once RACING starts, traffic detection is never used again.

GREEN_CONFIRM_FRAMES = 3

# Normalized camera search ROI: x1, y1, x2, y2
# 본선에서는 신호등 위치가 랜덤일 수 있으므로 고정 우측 ROI 대신
# 화면 대부분을 탐색한다. 하단 12%는 도로/차선 오탐을 줄이기 위해 제외.
TRAFFIC_ROI_NORM = (0.02, 0.02, 0.98, 0.88)

# ------------------------------------------------------------
# Pan / Tilt traffic-light search
# ------------------------------------------------------------
# test_v7 차선 ROI는 기본 카메라 자세에서 검증되었으므로 주행 자세는 0/0을 유지한다.
# 실차에서 별도 주행 tilt가 필요하다고 확인되면 DRIVE_CAMERA_TILT_DEG만 조정한다.
DRIVE_CAMERA_PAN_DEG = 0.0
DRIVE_CAMERA_TILT_DEG = 0.0

# 우선 현재 정면 시야에서 찾고, 없을 때만 좌/우 및 위쪽을 순차 탐색.
# 카메라 pan/tilt 하드웨어 범위(±30 deg) 안에서만 명령한다.
TRAFFIC_SEARCH_POSES = [
    (0.0, 0.0),
    (-20.0, 0.0),
    (20.0, 0.0),
    (-30.0, 0.0),
    (30.0, 0.0),
    (0.0, 15.0),
    (-20.0, 15.0),
    (20.0, 15.0),
    (-30.0, 15.0),
    (30.0, 15.0),
]

# 현재 버전은 HTTP pan/tilt 명령 후 짧은 정착시간을 둔다.
# /joint_states 기반 실제 각도 확인은 후속 실차 검증에서 필요 시 추가한다.
CAMERA_SETTLE_SEC = 0.30
TRAFFIC_POSE_DWELL_SEC = 0.60
TRAFFIC_LOCK_FRAMES = 2
TRAFFIC_LOCK_LOST_SEC = 1.20

# OpenCV HSV: H=[0,179], S/V=[0,255]
TRAFFIC_RED_H1 = (0, 12)
TRAFFIC_RED_H2 = (168, 179)
TRAFFIC_RED_S_MIN = 145
TRAFFIC_RED_V_MIN = 155

TRAFFIC_GREEN_H = (38, 92)
TRAFFIC_GREEN_S_MIN = 120
TRAFFIC_GREEN_V_MIN = 150

# BGR channel dominance
TRAFFIC_RED_CHANNEL_MIN = 135
TRAFFIC_RED_DOMINANCE = 45
TRAFFIC_GREEN_CHANNEL_MIN = 135
TRAFFIC_GREEN_DOMINANCE = 35

# Blob/context filters
TRAFFIC_MIN_BLOB_AREA_RATIO = 0.00005
TRAFFIC_MAX_BLOB_AREA_RATIO = 0.080
TRAFFIC_MIN_BBOX_FILL = 0.38
TRAFFIC_MIN_CIRCULARITY = 0.28
TRAFFIC_EDGE_MARGIN_PX = 3

# Dark background around illuminated lamp
TRAFFIC_DARK_V_MAX = 95
TRAFFIC_DARK_EXPAND = 2.1
TRAFFIC_MIN_DARK_SURROUND_RATIO = 0.25
TRAFFIC_MIN_RING_PIXELS = 30

TRAFFIC_SCORE_MIN = 0.50
TRAFFIC_GREEN_OVER_RED_MARGIN = 0.08

# ============================================================
# 4-B. SHARP-CORNER FALLBACK
# ============================================================
# NORMAL driving still uses WHITE outer boundaries.
# Yellow is used ONLY as a short-lived path cue when a ~90 degree
# corner makes x=f(y) white-boundary fitting geometrically invalid.

YELLOW_H_MIN = 5
YELLOW_H_MAX = 35
YELLOW_S_MIN = 110
YELLOW_V_MIN = 100

YELLOW_MIN_AREA = 12
YELLOW_MAX_AREA = 5000

# Require yellow candidates to be on/next to dark asphalt.
ASPHALT_V_MAX = 175
ASPHALT_DILATE = 17

CORNER_ENTER_HEADING = 0.13
CORNER_ENTER_STEER = 9.0
CORNER_EXIT_HEADING = 0.085
CORNER_EXIT_STEER = 6.0
CORNER_EXIT_BOTH_FRAMES = 4

CORNER_LOOKAHEAD_PX = 125.0
CORNER_MAX_LINK_PX = 185.0
CORNER_FIRST_LINK_PX = 190.0
CORNER_BACKWARD_ALLOW_PX = 35.0

# Image-space target angle -> wheel steering.
# 45 deg target direction becomes ~18 deg steering.
CORNER_STEER_GAIN = 0.40

# If the yellow path disappears for a few frames in the middle of the turn,
# keep the last corner steering briefly instead of stopping instantly.
CORNER_HOLD_FRAMES = 7

# V9.4 safe white-lane fallback after yellow corner path is lost.
# Raw one-line steering can jump directly to +/-20 deg, so do not switch
# to it in one frame. Move only a few degrees per camera frame.
CORNER_FALLBACK_STEER_STEP_DEG = 3.0
SPEED_CORNER_FALLBACK = 0.40

# ============================================================
# 5. PhysiCar HTTP API
# ============================================================

def camera():
    """Return latest camera frame as BGR ndarray."""
    response = session.get(f"{BASE_URL}/camera", timeout=2)
    response.raise_for_status()

    img = cv2.imdecode(
        np.frombuffer(response.content, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise RuntimeError("camera image decode failed")

    return img


def drive(speed_mps, steering_deg):
    """
    PhysiCar example convention:
      speed: m/s
      steering_deg: degrees in this program
      /steering endpoint receives radians
      positive steering = left
    """
    r1 = session.post(
        f"{BASE_URL}/speed",
        json={"value": float(speed_mps)},
        timeout=2,
    )
    r1.raise_for_status()

    r2 = session.post(
        f"{BASE_URL}/steering",
        json={"value": math.radians(float(steering_deg))},
        timeout=2,
    )
    r2.raise_for_status()


def stop_vehicle():
    try:
        drive(0.0, 0.0)
    except Exception:
        pass


def set_camera_pose(pan_deg, tilt_deg):
    """Command camera pan/tilt in degrees. Returns True on success."""
    try:
        pan_deg = float(np.clip(pan_deg, -30.0, 30.0))
        tilt_deg = float(np.clip(tilt_deg, -30.0, 30.0))

        r1 = session.post(
            f"{BASE_URL}/camera/pan",
            json={"value": math.radians(pan_deg)},
            timeout=2,
        )
        r1.raise_for_status()

        r2 = session.post(
            f"{BASE_URL}/camera/tilt",
            json={"value": math.radians(tilt_deg)},
            timeout=2,
        )
        r2.raise_for_status()

        return True

    except Exception as e:
        print(f"[CAMERA_POSE] {type(e).__name__}: {e}")
        return False


# ============================================================
# 5-B. TRAFFIC LIGHT PERCEPTION
# ============================================================

def traffic_search_roi(frame):
    h, w = frame.shape[:2]
    x1n, y1n, x2n, y2n = TRAFFIC_ROI_NORM

    x1 = int(np.clip(x1n * w, 0, w - 1))
    y1 = int(np.clip(y1n * h, 0, h - 1))
    x2 = int(np.clip(x2n * w, x1 + 1, w))
    y2 = int(np.clip(y2n * h, y1 + 1, h))

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def traffic_make_masks(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(roi)

    # ---------- RED ----------
    red1 = cv2.inRange(
        hsv,
        np.array([
            TRAFFIC_RED_H1[0],
            TRAFFIC_RED_S_MIN,
            TRAFFIC_RED_V_MIN,
        ], dtype=np.uint8),
        np.array([
            TRAFFIC_RED_H1[1],
            255,
            255,
        ], dtype=np.uint8),
    )

    red2 = cv2.inRange(
        hsv,
        np.array([
            TRAFFIC_RED_H2[0],
            TRAFFIC_RED_S_MIN,
            TRAFFIC_RED_V_MIN,
        ], dtype=np.uint8),
        np.array([
            TRAFFIC_RED_H2[1],
            255,
            255,
        ], dtype=np.uint8),
    )

    red_hsv = cv2.bitwise_or(red1, red2)

    ri = r.astype(np.int16)
    gi = g.astype(np.int16)
    bi = b.astype(np.int16)

    red_dom = (
        (ri >= TRAFFIC_RED_CHANNEL_MIN)
        & ((ri - gi) >= TRAFFIC_RED_DOMINANCE)
        & ((ri - bi) >= TRAFFIC_RED_DOMINANCE)
    ).astype(np.uint8) * 255

    red = cv2.bitwise_and(red_hsv, red_dom)

    # ---------- GREEN ----------
    green_hsv = cv2.inRange(
        hsv,
        np.array([
            TRAFFIC_GREEN_H[0],
            TRAFFIC_GREEN_S_MIN,
            TRAFFIC_GREEN_V_MIN,
        ], dtype=np.uint8),
        np.array([
            TRAFFIC_GREEN_H[1],
            255,
            255,
        ], dtype=np.uint8),
    )

    green_dom = (
        (gi >= TRAFFIC_GREEN_CHANNEL_MIN)
        & ((gi - ri) >= TRAFFIC_GREEN_DOMINANCE)
        & ((gi - bi) >= TRAFFIC_GREEN_DOMINANCE)
    ).astype(np.uint8) * 255

    green = cv2.bitwise_and(green_hsv, green_dom)

    # Preserve small illuminated lamps.
    k3 = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k3)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k3)

    return red, green, hsv


def traffic_dark_surround_ratio(hsv, contour):
    h, w = hsv.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)

    cx = x + bw / 2.0
    cy = y + bh / 2.0

    ew = max(bw + 4, int(round(bw * TRAFFIC_DARK_EXPAND)))
    eh = max(bh + 4, int(round(bh * TRAFFIC_DARK_EXPAND)))

    ex1 = max(0, int(round(cx - ew / 2)))
    ey1 = max(0, int(round(cy - eh / 2)))
    ex2 = min(w, int(round(cx + ew / 2)))
    ey2 = min(h, int(round(cy + eh / 2)))

    if ex2 <= ex1 or ey2 <= ey1:
        return 0.0

    ring = np.full(
        (ey2 - ey1, ex2 - ex1),
        255,
        dtype=np.uint8,
    )

    shifted = contour.copy()
    shifted[:, 0, 0] -= ex1
    shifted[:, 0, 1] -= ey1

    cv2.drawContours(
        ring,
        [shifted],
        -1,
        0,
        thickness=-1,
    )

    v = hsv[ey1:ey2, ex1:ex2, 2]
    valid = ring > 0
    n = int(np.count_nonzero(valid))

    if n < TRAFFIC_MIN_RING_PIXELS:
        return 0.0

    dark = (v <= TRAFFIC_DARK_V_MAX) & valid

    return (
        float(np.count_nonzero(dark))
        / float(n)
    )


def traffic_candidate_score(
    area_ratio,
    fill,
    circularity,
    dark_ratio,
):
    area_score = float(np.clip(
        (
            area_ratio
            - TRAFFIC_MIN_BLOB_AREA_RATIO
        )
        / max(
            0.0025
            - TRAFFIC_MIN_BLOB_AREA_RATIO,
            1e-6,
        ),
        0.0,
        1.0,
    ))

    fill_score = float(np.clip(
        (
            fill
            - TRAFFIC_MIN_BBOX_FILL
        )
        / max(
            0.85
            - TRAFFIC_MIN_BBOX_FILL,
            1e-6,
        ),
        0.0,
        1.0,
    ))

    circularity_score = float(np.clip(
        (
            circularity
            - TRAFFIC_MIN_CIRCULARITY
        )
        / max(
            0.90
            - TRAFFIC_MIN_CIRCULARITY,
            1e-6,
        ),
        0.0,
        1.0,
    ))

    dark_score = float(np.clip(
        (
            dark_ratio
            - TRAFFIC_MIN_DARK_SURROUND_RATIO
        )
        / max(
            0.90
            - TRAFFIC_MIN_DARK_SURROUND_RATIO,
            1e-6,
        ),
        0.0,
        1.0,
    ))

    return (
        0.15 * area_score
        + 0.20 * fill_score
        + 0.15 * circularity_score
        + 0.50 * dark_score
    )


def traffic_find_best_candidate(mask, hsv):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = mask.shape
    roi_area = float(h * w)

    best = None

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area <= 0.0:
            continue

        area_ratio = area / roi_area

        if not (
            TRAFFIC_MIN_BLOB_AREA_RATIO
            <= area_ratio
            <= TRAFFIC_MAX_BLOB_AREA_RATIO
        ):
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        if (
            x <= TRAFFIC_EDGE_MARGIN_PX
            or y <= TRAFFIC_EDGE_MARGIN_PX
            or x + bw >= w - TRAFFIC_EDGE_MARGIN_PX
            or y + bh >= h - TRAFFIC_EDGE_MARGIN_PX
        ):
            continue

        bbox_area = float(max(1, bw * bh))
        fill = area / bbox_area

        if fill < TRAFFIC_MIN_BBOX_FILL:
            continue

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        if perimeter <= 1e-6:
            continue

        circularity = float(
            4.0 * np.pi * area
            / (perimeter * perimeter)
        )

        if circularity < TRAFFIC_MIN_CIRCULARITY:
            continue

        dark_ratio = traffic_dark_surround_ratio(
            hsv,
            contour,
        )

        if (
            dark_ratio
            < TRAFFIC_MIN_DARK_SURROUND_RATIO
        ):
            continue

        score = traffic_candidate_score(
            area_ratio,
            fill,
            circularity,
            dark_ratio,
        )

        candidate = {
            "contour": contour,
            "bbox": (x, y, bw, bh),
            "area_ratio": area_ratio,
            "fill": fill,
            "circularity": circularity,
            "dark_ratio": dark_ratio,
            "score": score,
        }

        if (
            best is None
            or candidate["score"] > best["score"]
        ):
            best = candidate

    return best


class TrafficDetector:
    def __init__(self):
        self.green_streak = 0

    def reset(self):
        self.green_streak = 0

    def update(
        self,
        red_candidate,
        green_candidate,
    ):
        red_score = (
            red_candidate["score"]
            if red_candidate is not None
            else 0.0
        )

        green_score = (
            green_candidate["score"]
            if green_candidate is not None
            else 0.0
        )

        red_valid = (
            red_score >= TRAFFIC_SCORE_MIN
        )

        green_valid = (
            green_score >= TRAFFIC_SCORE_MIN
            and green_score
            >= red_score
            + TRAFFIC_GREEN_OVER_RED_MARGIN
        )

        # Safety priority: credible RED always blocks start.
        if red_valid:
            raw_state = "RED"
            self.green_streak = 0

        elif green_valid:
            raw_state = "GREEN"
            self.green_streak += 1

        else:
            raw_state = "UNKNOWN"
            self.green_streak = 0

        confirmed_green = (
            self.green_streak
            >= GREEN_CONFIRM_FRAMES
        )

        return {
            "raw_state": raw_state,
            "confirmed_green": confirmed_green,
            "green_streak": self.green_streak,
            "red_score": red_score,
            "green_score": green_score,
        }


def traffic_draw_candidate(
    image,
    candidate,
    color,
    label,
):
    if candidate is None:
        return

    x, y, w, h = candidate["bbox"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    cv2.putText(
        image,
        f"{label} {candidate['score']:.2f}",
        (x, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )


def make_traffic_wait_panel(
    frame,
    roi,
    roi_box,
    red_mask,
    green_mask,
    red_candidate,
    green_candidate,
    result,
):
    h, w = frame.shape[:2]

    camera_vis = frame.copy()

    x1, y1, x2, y2 = roi_box
    cv2.rectangle(
        camera_vis,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        camera_vis,
        "WAIT_GREEN - traffic search ROI",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    roi_vis = roi.copy()

    traffic_draw_candidate(
        roi_vis,
        red_candidate,
        (0, 0, 255),
        "RED",
    )

    traffic_draw_candidate(
        roi_vis,
        green_candidate,
        (0, 255, 0),
        "GREEN",
    )

    cv2.putText(
        roi_vis,
        "TRAFFIC ROI",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    red_vis = cv2.cvtColor(
        red_mask,
        cv2.COLOR_GRAY2BGR,
    )

    green_vis = cv2.cvtColor(
        green_mask,
        cv2.COLOR_GRAY2BGR,
    )

    # Lower-left: red/green masks side-by-side.
    half = max(1, w // 2)

    red_vis = cv2.resize(
        red_vis,
        (half, h),
        interpolation=cv2.INTER_NEAREST,
    )

    green_vis = cv2.resize(
        green_vis,
        (w - half, h),
        interpolation=cv2.INTER_NEAREST,
    )

    masks = np.hstack([
        red_vis,
        green_vis,
    ])

    cv2.putText(
        masks,
        "RED MASK",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        masks,
        "GREEN MASK",
        (half + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    info = np.zeros_like(frame)

    raw = result["raw_state"]

    if raw == "RED":
        headline = "RED - WAIT"
        color = (0, 0, 255)

    elif raw == "GREEN":
        headline = "GREEN - CONFIRMING"
        color = (0, 255, 0)

    else:
        headline = "UNKNOWN - WAIT"
        color = (255, 255, 255)

    lines = [
        headline,
        (
            f"green streak: "
            f"{result['green_streak']}/"
            f"{GREEN_CONFIRM_FRAMES}"
        ),
        f"red score: {result['red_score']:.3f}",
        f"green score: {result['green_score']:.3f}",
        "vehicle: STOPPED",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            info,
            line,
            (18, 42 + i * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 if i == 0 else 0.58,
            color if i == 0 else (235, 235, 235),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )

    # Make ROI view same size as camera panel.
    roi_big = cv2.resize(
        roi_vis,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    top = np.hstack([
        camera_vis,
        roi_big,
    ])

    bottom = np.hstack([
        masks,
        info,
    ])

    return np.vstack([
        top,
        bottom,
    ])

# ============================================================
# 5-C. RANDOM-POSITION TRAFFIC SEARCH FSM
# ============================================================

def _traffic_detect_frame(img, detector):
    """Run the existing test_v7 traffic detector on one camera frame."""
    traffic_roi, traffic_roi_box = traffic_search_roi(img)
    traffic_red_mask, traffic_green_mask, traffic_hsv = traffic_make_masks(
        traffic_roi
    )

    red_candidate = traffic_find_best_candidate(
        traffic_red_mask,
        traffic_hsv,
    )
    green_candidate = traffic_find_best_candidate(
        traffic_green_mask,
        traffic_hsv,
    )

    result = detector.update(
        red_candidate,
        green_candidate,
    )

    panel = make_traffic_wait_panel(
        img,
        traffic_roi,
        traffic_roi_box,
        traffic_red_mask,
        traffic_green_mask,
        red_candidate,
        green_candidate,
        result,
    )

    red_pixels = int(np.count_nonzero(traffic_red_mask))
    green_pixels = int(np.count_nonzero(traffic_green_mask))

    return {
        "result": result,
        "panel": panel,
        "red_pixels": red_pixels,
        "green_pixels": green_pixels,
        "red_candidate": red_candidate,
        "green_candidate": green_candidate,
    }


def _traffic_print(prefix, pan_deg, tilt_deg, pack):
    result = pack["result"]
    print(
        f"[{prefix}] "
        f"pan={pan_deg:+.0f} tilt={tilt_deg:+.0f} "
        f"state={result['raw_state']:<7} "
        f"green={result['green_streak']}/{GREEN_CONFIRM_FRAMES} "
        f"red={result['red_score']:.3f} "
        f"green_score={result['green_score']:.3f} "
        f"rpix={pack['red_pixels']:<5d} "
        f"gpix={pack['green_pixels']:<5d}"
    )


def _wait_green_at_locked_pose(detector, pan_deg, tilt_deg):
    """
    Camera stays fixed at the pose where a credible lamp was found.
    Returns True after confirmed GREEN.
    Returns False if the lamp disappears for too long, so global search resumes.
    """
    last_seen = time.time()
    last_print = 0.0

    print(
        f"[TRAFFIC] LOCKED at pan={pan_deg:+.1f} deg, "
        f"tilt={tilt_deg:+.1f} deg -> WAIT_GREEN"
    )

    while True:
        loop_start = time.time()

        try:
            img = camera()
        except Exception as e:
            print(f"[TRAFFIC] camera error while locked: {e}")
            if DRIVE_ENABLED:
                stop_vehicle()
            time.sleep(0.10)
            continue

        pack = _traffic_detect_frame(img, detector)
        result = pack["result"]

        if result["raw_state"] != "UNKNOWN":
            last_seen = time.time()

        status = (
            f"WAIT_GREEN LOCKED pan={pan_deg:+.0f} tilt={tilt_deg:+.0f} "
            f"state={result['raw_state']} "
            f"green={result['green_streak']}/{GREEN_CONFIRM_FRAMES} "
            f"red={result['red_score']:.3f} "
            f"green_score={result['green_score']:.3f}"
        )
        update_web(pack["panel"], status)

        now = time.time()
        if now - last_print >= PRINT_INTERVAL:
            _traffic_print("TRAFFIC LOCK", pan_deg, tilt_deg, pack)
            last_print = now

        if result["confirmed_green"]:
            print(
                f"[TRAFFIC] GREEN CONFIRMED ({GREEN_CONFIRM_FRAMES} frames) "
                f"at pan={pan_deg:+.1f}, tilt={tilt_deg:+.1f}"
            )
            return True

        # Do not unlock on one or two UNKNOWN frames. Only resume search if
        # the previously locked lamp has really disappeared for a while.
        if now - last_seen >= TRAFFIC_LOCK_LOST_SEC:
            print(
                f"[TRAFFIC] locked light lost for {TRAFFIC_LOCK_LOST_SEC:.1f}s "
                "-> resume SEARCH"
            )
            detector.reset()
            return False

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, 1.0 / TARGET_FPS - elapsed))


def search_and_wait_for_green():
    """
    Pre-race traffic-light controller.

    SEARCH:
      - Vehicle remains stopped.
      - First inspect the current forward view.
      - If no credible RED/GREEN lamp is found, scan pan/tilt poses.

    LOCK / WAIT_GREEN:
      - Once RED/GREEN is seen for TRAFFIC_LOCK_FRAMES, stop scanning.
      - Keep the camera fixed at that pose and wait for GREEN confirmation.

    RETURN_CAMERA:
      - After GREEN latch, restore the camera to the proven test_v7 drive pose.
      - Only then return to the lane-driving loop.

    There is intentionally NO timeout that forces the vehicle to start.
    """
    detector = TrafficDetector()
    pose_index = 0

    if DRIVE_ENABLED:
        stop_vehicle()

    print("[TRAFFIC] SEARCH_LIGHT start; vehicle remains STOPPED")

    while True:
        pan_deg, tilt_deg = TRAFFIC_SEARCH_POSES[pose_index]

        print(
            f"[TRAFFIC] move camera -> pose {pose_index + 1}/"
            f"{len(TRAFFIC_SEARCH_POSES)} "
            f"pan={pan_deg:+.1f}, tilt={tilt_deg:+.1f}"
        )

        set_camera_pose(pan_deg, tilt_deg)
        detector.reset()
        time.sleep(CAMERA_SETTLE_SEC)

        detect_started = time.time()
        lock_streak = 0
        lock_state = None
        last_print = 0.0
        locked = False

        while time.time() - detect_started < TRAFFIC_POSE_DWELL_SEC:
            loop_start = time.time()

            try:
                img = camera()
            except Exception as e:
                print(f"[TRAFFIC] camera error during search: {e}")
                if DRIVE_ENABLED:
                    stop_vehicle()
                time.sleep(0.10)
                continue

            pack = _traffic_detect_frame(img, detector)
            result = pack["result"]
            raw_state = result["raw_state"]

            if raw_state != "UNKNOWN":
                if raw_state == lock_state:
                    lock_streak += 1
                else:
                    lock_state = raw_state
                    lock_streak = 1
            else:
                lock_state = None
                lock_streak = 0

            status = (
                f"SEARCH_LIGHT pose={pose_index + 1}/{len(TRAFFIC_SEARCH_POSES)} "
                f"pan={pan_deg:+.0f} tilt={tilt_deg:+.0f} "
                f"state={raw_state} lock={lock_streak}/{TRAFFIC_LOCK_FRAMES} "
                f"red={result['red_score']:.3f} "
                f"green_score={result['green_score']:.3f}"
            )
            update_web(pack["panel"], status)

            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                _traffic_print("TRAFFIC SEARCH", pan_deg, tilt_deg, pack)
                last_print = now

            if lock_streak >= TRAFFIC_LOCK_FRAMES:
                locked = True
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, 1.0 / TARGET_FPS - elapsed))

        if locked:
            green = _wait_green_at_locked_pose(
                detector,
                pan_deg,
                tilt_deg,
            )

            if green:
                print(
                    "[TRAFFIC] RETURN_CAMERA -> "
                    f"pan={DRIVE_CAMERA_PAN_DEG:+.1f}, "
                    f"tilt={DRIVE_CAMERA_TILT_DEG:+.1f}"
                )
                set_camera_pose(
                    DRIVE_CAMERA_PAN_DEG,
                    DRIVE_CAMERA_TILT_DEG,
                )

                # Keep vehicle stopped while the camera physically returns.
                if DRIVE_ENABLED:
                    stop_vehicle()
                time.sleep(CAMERA_SETTLE_SEC)

                print("[TRAFFIC] camera returned -> RACING enabled")
                return

        # No lamp at this pose, or a locked lamp disappeared.
        # Continue cyclic search. Never start just because search took too long.
        pose_index = (pose_index + 1) % len(TRAFFIC_SEARCH_POSES)


# ============================================================
# 5-D. LiDAR READ-ONLY CLUSTER LOGGER
# ============================================================
# IMPORTANT:
# - /scan callback은 최신 scan만 저장합니다.
# - cluster 계산은 별도 worker thread에서 5 Hz로만 수행합니다.
# - 차량 제어값(speed/steering)에는 절대 연결하지 않습니다.
# - 0 deg = 차량 정면 (시뮬레이터에서 확인 완료)
# - +angle = LEFT, -angle = RIGHT

LIDAR_TOPIC = "/scan"

# 기존 주행에 부담을 최소화하기 위해 cluster 분석은 5 Hz.
LIDAR_CLUSTER_HZ = 10.0

# 분석 범위. 차량 앞/옆으로 지나가는 물체를 보기 위해 ±100 deg.
LIDAR_CLUSTER_FOV_DEG = 100.0

# 너무 먼 구조물은 이번 분석에서 관심 없음.
LIDAR_CLUSTER_MAX_RANGE_M = 2.0

# contiguous cluster 조건
# 인접 LiDAR point의 XY 거리 차이가 이 값보다 크면 cluster를 분리.
LIDAR_CLUSTER_LINK_M = 0.12

# 1 point noise 제외
LIDAR_CLUSTER_MIN_POINTS = 3

# 출력할 cluster 수
LIDAR_CLUSTER_MAX_PRINT = 8

# ============================================================
# READ-ONLY CONE DETECTOR
# ============================================================
# 이 값들은 방금 주행 로그에서 관측된 라바콘/벽 cluster 특성에 맞춘
# '첫 검증용' 값입니다. 아직 차량 제어에는 사용하지 않습니다.

# 라바콘은 실제 XY 폭이 약 0.10~0.14 m로 반복 관측됨.
CONE_WIDTH_MIN_M = 0.07
CONE_WIDTH_MAX_M = 0.20

# 1~2 point noise 제외
CONE_MIN_POINTS = 5

# 가까이 붙은 라바콘은 angular span이 커지므로 너무 작게 제한하지 않음.
CONE_MAX_ANGLE_SPAN_DEG = 30.0

# 최초 탐지 거리 / 각도
CONE_DETECT_MAX_RANGE_M = 1.50
CONE_SEARCH_ANGLE_DEG = 35.0

# 최초 탐지 시 차량 중심에서 너무 옆에 있는 작은 구조물은 제외.
# 실제 라바콘은 접근 시작 시 y가 대체로 약 ±0.2 m 근처였음.
CONE_SEARCH_MAX_ABS_Y_M = 0.55

# 2번 연속 같은 compact object가 관측되어야 DETECTED.
CONE_CONFIRM_FRAMES = 2

# LOCK 이후에는 옆으로 지나가는 과정까지 추적.
CONE_TRACK_MAX_RANGE_M = 1.80
CONE_TRACK_ANGLE_DEG = 100.0

# 이전 cluster 중심과의 최대 이동량.
# 5 Hz, 최고속도 0.65 m/s에서 충분한 여유를 둠.
CONE_TRACK_MAX_CENTER_SHIFT_M = 0.45

# PASSING 판정 보조 기준
CONE_PASSING_ANGLE_DEG = 45.0

# LOCK 이후 몇 번 연속 못 찾으면 해당 물체 추적 종료
CONE_LOST_FRAMES = 3

# ============================================================
# CORNER-SPECIFIC CONE ACQUISITION / PASSING
# ============================================================
# In a bend, cones appear at a much larger bearing than on a straight road.
# Keep the normal detector conservative, but widen acquisition only while the
# lane/corner geometry says we are entering or staying in a corner.
CONE_CORNER_DETECT_MAX_RANGE_M = 1.95
CONE_CORNER_SEARCH_ANGLE_DEG = 75.0
CONE_CORNER_SEARCH_MAX_ABS_Y_M = 0.85
# Farther cones can return only a few LiDAR rays and their measured XY width
# can be smaller than the close-range 0.07 m cone width. Relax ONLY the
# corner-acquisition gate; the normal straight detector stays unchanged.
CONE_CORNER_WIDTH_MIN_M = 0.025
CONE_CORNER_WIDTH_MAX_M = 0.26
CONE_CORNER_MAX_ANGLE_SPAN_DEG = 40.0
CONE_CORNER_MIN_POINTS = 2

# If a plausible cone is already this close in a corner, waiting for the
# second 5-Hz confirmation costs too much distance. Lock it immediately.
CONE_CORNER_URGENT_RANGE_M = 1.20

# IMPORTANT: angle is not a reliable passing cue in a corner because ego yaw
# itself can push a still-ahead cone beyond +/-45 deg. In a corner, declare
# PASSING only after the cone reaches the vehicle-side x position.
CONE_CORNER_PASS_X_M = 0.08
CONE_CORNER_TRACK_MAX_CENTER_SHIFT_M = 0.65

# Independent last-resort LiDAR collision bubble. This does not need the
# normal cone confirmation FSM; it only reacts to a compact object physically
# entering the narrow vehicle corridor immediately ahead.
RAW_SHIELD_MAX_WIDTH_M = 0.32
RAW_SHIELD_MIN_POINTS = 2

# V14 early-acquire + escape-creep:
# Do NOT decide collision risk from the cluster center |cy| alone.
# A cone can have cy~=0.25 m while already safely beside the vehicle.
# Use the cluster's nearest lateral edge (y_min/y_max) instead.
#
# hard corridor : definite vehicle-footprint overlap -> emergency stop allowed
# soft corridor : near-miss margin -> crawl only
RAW_SHIELD_HARD_HALF_WIDTH_M = 0.17
RAW_SHIELD_SOFT_HALF_WIDTH_M = 0.21
RAW_SHIELD_BRAKE_X_M = 0.70
RAW_SHIELD_STOP_X_M = 0.34
RAW_SHIELD_SIDE_PASS_X_M = 0.12
RAW_SHIELD_CRAWL_SPEED = 0.10
# A full stop at x~=0.30 m deadlocks a non-holonomic car: it cannot create
# lateral separation while stationary. If a validated corner escape path is
# already steering away from the obstacle, crawl through the manoeuvre and
# reserve a hard stop for truly immediate overlap.
RAW_SHIELD_ESCAPE_CRAWL_SPEED = 0.08
RAW_SHIELD_ABSOLUTE_STOP_X_M = 0.12

# ============================================================
# 5-E. V9 CONE AVOIDANCE
# ============================================================
# IMPORTANT
# - 회피 입력은 위 CONE detector 결과만 사용합니다.
# - wall_like cluster는 회피 FSM으로 들어오지 않습니다.
# - 회피 방향은 DETECTED 순간 1회 결정한 뒤 PASSED까지 고정합니다.
# - 기존 V8 lane/corner controller는 유지하고, lane target offset만 얹습니다.

AVOIDANCE_ENABLED = True

# Lane-center offset as a ratio of remembered lane width.
# 약 0.28 lane width까지 옆으로 이동시키는 첫 실주행용 보수 설정.
AVOID_FULL_OFFSET_RATIO = 0.24
AVOID_SHIFT_NEAR_RATIO = 0.06

# Offset target smoothing, applied at camera loop rate.
AVOID_OFFSET_ALPHA = 0.18

# Cone relative geometry thresholds.
AVOID_CENTER_DEADBAND_M = 0.03
AVOID_PASS_X_M = 0.10
AVOID_SAFE_LATERAL_M = 0.27

# Speed caps [m/s].
AVOID_SHIFT_SPEED = 0.32
AVOID_PASS_SPEED = 0.30
AVOID_RETURN_SPEED = 0.34
AVOID_RECOVER_SPEED = 0.40

# V9.3:
# 회피 자체 때문에 0.18~0.28 m/s까지 떨어지지 않도록
# active avoidance 전체에서 기존 빠른 주행 속도 0.65 m/s를 유지한다.
# NORMAL 직선에서는 SPEED_MAX=0.75까지 허용한다.
AVOID_CRUISE_SPEED = 0.65

# Late / very-close detection safety.
AVOID_LATE_X_M = 0.80
AVOID_LATE_SPEED = 0.26
AVOID_EMERGENCY_X_M = 0.42
AVOID_EMERGENCY_LATERAL_M = 0.22
AVOID_EMERGENCY_SPEED = 0.18

# In a sharp corner, preserve the corner controller and add only
# a limited avoidance steering correction.
AVOID_CORNER_DELTA_MAX_DEG = 8.0

# RECOVER -> NORMAL condition.
AVOID_OFFSET_DONE_RATIO = 0.018

# V9.2 direction selection:
# If the proven V8 controller was already in a strong corner immediately
# before cone confirmation, follow the route turn instead of blindly
# choosing the side opposite to the cone.
AVOID_ROUTE_STEER_MIN_DEG = 8.0
AVOID_ROUTE_OVERRIDE_MIN_LATERAL_M = 0.30

# ============================================================
# 5-F. CURVED-PATH / DRIVABLE-ROAD GUARD
# ============================================================
# Straight sections keep the original V9 lane-offset avoidance.
# In an active sharp corner, however, a global BEV x-offset is geometrically
# wrong because the road's lateral direction rotates with the bend.
#
# Therefore corner avoidance is applied along the local normal of the
# detected yellow corner path (Frenet-like lateral shift), then the shifted
# target is checked against a drivable asphalt mask. If the requested shift
# leaves the road, it is reduced until the target is safely inside.
ROAD_GUARD_ENABLED = True
ROAD_GUARD_CLOSE_KERNEL = 19
ROAD_GUARD_OPEN_KERNEL = 5
ROAD_GUARD_MIN_COMPONENT_RATIO = 0.08
ROAD_GUARD_MARGIN_LANE_RATIO = 0.16
ROAD_GUARD_MIN_MARGIN_PX = 8.0
ROAD_GUARD_SHIFT_STEPS = 12

# Corner + obstacle is deliberately slower than the old unconditional
# AVOID_CRUISE_SPEED=0.65 m/s. This is only a safety cap while the curved
# avoidance target is active; straight avoidance keeps the original speed.
AVOID_CORNER_GUARD_SPEED = 0.38

# Distance-scaled corner shift. Far cones should not trigger an abrupt large
# lateral jump on a bend; the requested path-normal shift grows as the cone
# approaches. This reduces corner path departure while retaining strong close
# avoidance.
CORNER_AVOID_SHIFT_FAR_RATIO = 0.08
CORNER_AVOID_SHIFT_NEAR_RATIO = 0.24
CORNER_AVOID_SHIFT_FAR_X_M = 1.40
CORNER_AVOID_SHIFT_NEAR_X_M = 0.45

# V13 dual-path corner planner.
# Build BOTH path-relative LEFT/RIGHT candidates and choose the one that is
# actually feasible inside the road. Obstacle-opposite side gets a preference,
# but feasibility wins when the preferred side has no room.
CORNER_DUALPATH_MIN_SCALE = 0.18
CORNER_DUALPATH_PREFERENCE_BONUS = 0.35
CORNER_DUALPATH_CONTINUITY_BONUS = 0.08
CORNER_REACQUIRE_X_M = 0.18

# Last-resort collision shield. If the road guard cannot provide enough safe
# lateral shift while a cone is still directly ahead, reduce speed sharply.
# At emergency distance, stop rather than drive the reference path into it.
CORNER_GUARD_MIN_USEFUL_SCALE = 0.22
CORNER_UNSAFE_BRAKE_X_M = 0.90
CORNER_UNSAFE_BRAKE_ABS_Y_M = 0.45
CORNER_UNSAFE_SPEED = 0.10
CORNER_EMERGENCY_STOP_X_M = 0.42
CORNER_EMERGENCY_STOP_ABS_Y_M = 0.30

_lidar_lock = threading.Lock()
_lidar_latest_scan = None
_lidar_latest_clusters = []
_lidar_node = None
_lidar_spin_thread = None
_lidar_worker_thread = None
_lidar_stop_event = threading.Event()
_lidar_started_rclpy = False


def _normalize_angle_deg(angle_deg):
    """Normalize angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


class _LidarReadOnlyNode(Node):
    def __init__(self):
        super().__init__("test_v8_lidar_cluster_logger")

        self.subscription = self.create_subscription(
            LaserScan,
            LIDAR_TOPIC,
            self._scan_callback,
            qos_profile_sensor_data,
        )

    def _scan_callback(self, msg):
        global _lidar_latest_scan

        ranges = np.asarray(msg.ranges, dtype=np.float32)

        if ranges.size == 0:
            return

        pack = {
            "ranges": ranges.copy(),
            "angle_min": float(msg.angle_min),
            "angle_increment": float(msg.angle_increment),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
            "stamp": time.monotonic(),
        }

        # callback에서는 저장만 한다.
        with _lidar_lock:
            _lidar_latest_scan = pack


def _build_lidar_clusters(scan):
    """
    Build simple contiguous LiDAR clusters from the latest scan.

    This is intentionally ANALYSIS-ONLY.
    No cone/wall decision is made here.

    Each returned cluster contains:
      n          : point count
      dmin       : nearest range [m]
      dmed       : median range [m]
      angle      : median relative angle [deg]
      angle_min/max/span
      width      : XY bounding-box diagonal [m]
      x_min/max  : forward extent [m]
      y_min/max  : lateral extent [m]
    """
    ranges = scan["ranges"]

    idx = np.arange(ranges.size, dtype=np.float32)
    angles_rad = (
        scan["angle_min"]
        + idx * scan["angle_increment"]
    )
    angles_deg = np.degrees(angles_rad)

    valid = (
        np.isfinite(ranges)
        & (ranges >= scan["range_min"])
        & (ranges <= min(
            scan["range_max"],
            LIDAR_CLUSTER_MAX_RANGE_M,
        ))
        & (angles_deg >= -LIDAR_CLUSTER_FOV_DEG)
        & (angles_deg <= +LIDAR_CLUSTER_FOV_DEG)
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

        # If there is a missing angular sample between valid points,
        # do not bridge the gap.
        index_gap = i - prev_i

        xy_gap = math.hypot(
            float(xs[i] - xs[prev_i]),
            float(ys[i] - ys[prev_i]),
        )

        if index_gap == 1 and xy_gap <= LIDAR_CLUSTER_LINK_M:
            current.append(i)
        else:
            if len(current) >= LIDAR_CLUSTER_MIN_POINTS:
                clusters_idx.append(current)
            current = [i]

        prev_i = i

    if len(current) >= LIDAR_CLUSTER_MIN_POINTS:
        clusters_idx.append(current)

    clusters = []

    for inds in clusters_idx:
        inds = np.asarray(inds, dtype=np.int32)

        rr = ranges[inds]
        aa = angles_deg[inds]
        xx = xs[inds]
        yy = ys[inds]

        x_span = float(np.max(xx) - np.min(xx))
        y_span = float(np.max(yy) - np.min(yy))

        width = float(math.hypot(x_span, y_span))

        x_min = float(np.min(xx))
        x_max = float(np.max(xx))
        y_min = float(np.min(yy))
        y_max = float(np.max(yy))

        cluster = {
            "n": int(inds.size),
            "dmin": float(np.min(rr)),
            "dmed": float(np.median(rr)),
            "angle": float(np.median(aa)),
            "angle_min": float(np.min(aa)),
            "angle_max": float(np.max(aa)),
            "angle_span": float(np.max(aa) - np.min(aa)),
            "width": width,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "cx": 0.5 * (x_min + x_max),
            "cy": 0.5 * (y_min + y_max),
        }

        clusters.append(cluster)

    # 가까운 cluster부터 출력
    clusters.sort(key=lambda c: c["dmin"])

    return clusters


def _lidar_worker():
    global _lidar_latest_clusters

    last_stamp = None
    period = 1.0 / max(LIDAR_CLUSTER_HZ, 1e-6)

    while not _lidar_stop_event.is_set():
        started = time.monotonic()

        with _lidar_lock:
            scan = (
                None
                if _lidar_latest_scan is None
                else {
                    "ranges": _lidar_latest_scan["ranges"].copy(),
                    "angle_min": _lidar_latest_scan["angle_min"],
                    "angle_increment": _lidar_latest_scan["angle_increment"],
                    "range_min": _lidar_latest_scan["range_min"],
                    "range_max": _lidar_latest_scan["range_max"],
                    "stamp": _lidar_latest_scan["stamp"],
                }
            )

        if scan is not None and scan["stamp"] != last_stamp:
            clusters = _build_lidar_clusters(scan)

            with _lidar_lock:
                _lidar_latest_clusters = clusters

            last_stamp = scan["stamp"]

        elapsed = time.monotonic() - started
        _lidar_stop_event.wait(max(0.0, period - elapsed))


def start_lidar_cluster_logger():
    global _lidar_node
    global _lidar_spin_thread
    global _lidar_worker_thread
    global _lidar_started_rclpy

    try:
        _lidar_stop_event.clear()

        if not rclpy.ok():
            rclpy.init(args=None)
            _lidar_started_rclpy = True

        _lidar_node = _LidarReadOnlyNode()

        _lidar_spin_thread = threading.Thread(
            target=_lidar_spin,
            daemon=True,
            name="test-v8-lidar-spin",
        )
        _lidar_spin_thread.start()

        _lidar_worker_thread = threading.Thread(
            target=_lidar_worker,
            daemon=True,
            name="test-v8-lidar-cluster-worker",
        )
        _lidar_worker_thread.start()

        print(
            f"[LIDAR] READ-ONLY cluster logger started: {LIDAR_TOPIC} "
            f"@ {LIDAR_CLUSTER_HZ:.1f} Hz "
            "(NO control authority)"
        )

    except Exception as e:
        print(
            f"[LIDAR] start failed: "
            f"{type(e).__name__}: {e}"
        )


def _lidar_spin():
    try:
        rclpy.spin(_lidar_node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except Exception as e:
        print(
            f"[LIDAR] spin error: "
            f"{type(e).__name__}: {e}"
        )


def stop_lidar_cluster_logger():
    global _lidar_node

    _lidar_stop_event.set()

    try:
        if _lidar_node is not None:
            _lidar_node.destroy_node()
            _lidar_node = None
    except Exception:
        pass

    try:
        if _lidar_started_rclpy and rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

    if (
        _lidar_worker_thread is not None
        and _lidar_worker_thread.is_alive()
    ):
        _lidar_worker_thread.join(timeout=1.0)

    if (
        _lidar_spin_thread is not None
        and _lidar_spin_thread.is_alive()
    ):
        _lidar_spin_thread.join(timeout=1.0)

    print("[LIDAR] READ-ONLY cluster logger stopped")


def get_lidar_clusters():
    with _lidar_lock:
        return [dict(c) for c in _lidar_latest_clusters]


def print_lidar_clusters():
    clusters = get_lidar_clusters()

    if not clusters:
        print("[LCLUSTER] none within ±100deg / 2.0m")
        return

    shown = clusters[:LIDAR_CLUSTER_MAX_PRINT]

    parts = []

    for i, c in enumerate(shown, start=1):
        side = (
            "L" if c["angle"] > 3.0
            else "R" if c["angle"] < -3.0
            else "C"
        )

        parts.append(
            f"C{i}[{side}] "
            f"d={c['dmin']:.2f}m "
            f"a={c['angle']:+.1f}deg "
            f"n={c['n']} "
            f"asp={c['angle_span']:.1f}deg "
            f"w={c['width']:.2f}m "
            f"x={c['x_min']:.2f}..{c['x_max']:.2f} "
            f"y={c['y_min']:.2f}..{c['y_max']:.2f}"
        )

    print(
        f"[LCLUSTER] total={len(clusters)} | "
        + " | ".join(parts)
    )


def _cone_compact_shape(c, corner_hint=False):
    """Compact-object test; slightly more tolerant only in a corner."""
    if corner_hint:
        return (
            c["n"] >= CONE_CORNER_MIN_POINTS
            and CONE_CORNER_WIDTH_MIN_M <= c["width"] <= CONE_CORNER_WIDTH_MAX_M
            and c["angle_span"] <= CONE_CORNER_MAX_ANGLE_SPAN_DEG
        )

    return (
        c["n"] >= CONE_MIN_POINTS
        and CONE_WIDTH_MIN_M <= c["width"] <= CONE_WIDTH_MAX_M
        and c["angle_span"] <= CONE_MAX_ANGLE_SPAN_DEG
    )


def _cone_search_candidate(c, corner_hint=False):
    """Conservative straight search; wider acquisition only in corners."""
    if corner_hint:
        return (
            _cone_compact_shape(c, corner_hint=True)
            and c["dmin"] <= CONE_CORNER_DETECT_MAX_RANGE_M
            and abs(c["angle"]) <= CONE_CORNER_SEARCH_ANGLE_DEG
            and c["x_max"] > 0.15
            and abs(c["cy"]) <= CONE_CORNER_SEARCH_MAX_ABS_Y_M
        )

    return (
        _cone_compact_shape(c, corner_hint=False)
        and c["dmin"] <= CONE_DETECT_MAX_RANGE_M
        and abs(c["angle"]) <= CONE_SEARCH_ANGLE_DEG
        and c["x_max"] > 0.25
        and abs(c["cy"]) <= CONE_SEARCH_MAX_ABS_Y_M
    )


def _cone_track_candidate(c):
    """After lock, tolerate oblique cone shape while it moves to the side."""
    return (
        _cone_compact_shape(c, corner_hint=True)
        and c["dmin"] <= CONE_TRACK_MAX_RANGE_M
        and abs(c["angle"]) <= CONE_TRACK_ANGLE_DEG
    )


def _cone_side(angle_deg):
    if angle_deg > 3.0:
        return "LEFT"
    if angle_deg < -3.0:
        return "RIGHT"
    return "CENTER"


def _cluster_center_distance(a, b):
    return math.hypot(
        float(a["cx"] - b["cx"]),
        float(a["cy"] - b["cy"]),
    )


class ConeReadOnlyDetector:
    """
    READ-ONLY detector.

    SEARCH -> CANDIDATE -> DETECTED -> PASSING -> NONE

    Nothing in this class can command vehicle speed or steering.
    """

    def __init__(self):
        self.candidate = None
        self.candidate_streak = 0

        self.locked = None
        self.lost_streak = 0

        self.last_state = "NONE"
        self.passed_latch = 0
        self.corner_hint_last = False

    def _best_search_candidate(self, clusters, corner_hint=False):
        candidates = [
            c for c in clusters
            if _cone_search_candidate(c, corner_hint=corner_hint)
        ]

        if not candidates:
            return None

        # 가장 가까운 전방 compact object 우선
        return min(
            candidates,
            key=lambda c: (
                c["dmin"],
                abs(c["angle"]),
            ),
        )

    def _match_previous(self, previous, clusters, corner_hint=False):
        if previous is None:
            return None

        candidates = [
            c for c in clusters
            if _cone_track_candidate(c)
        ]

        if not candidates:
            return None

        scored = []

        for c in candidates:
            center_shift = _cluster_center_distance(
                previous,
                c,
            )

            max_shift = (
                CONE_CORNER_TRACK_MAX_CENTER_SHIFT_M
                if corner_hint
                else CONE_TRACK_MAX_CENTER_SHIFT_M
            )
            if center_shift > max_shift:
                continue

            # 중심 이동량을 가장 중요하게 보고,
            # 폭 변화/각도 변화는 작은 보조항으로 사용.
            score = (
                center_shift
                + 0.30 * abs(c["width"] - previous["width"])
                + 0.002 * abs(c["angle"] - previous["angle"])
            )

            scored.append((score, c))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def update(self, clusters, corner_hint=False):
        self.corner_hint_last = bool(corner_hint)

        # PASSED는 한 번만 보여주고 다음 cycle에서 NONE으로 복귀.
        if self.passed_latch > 0:
            self.passed_latch -= 1

        # ----------------------------------------------------
        # LOCKED / TRACKING
        # ----------------------------------------------------
        if self.locked is not None:
            match = self._match_previous(
                self.locked,
                clusters,
                corner_hint=self.corner_hint_last,
            )

            if match is not None:
                self.locked = match
                self.lost_streak = 0

                side = _cone_side(match["angle"])

                if self.corner_hint_last:
                    passing = float(match["cx"]) <= CONE_CORNER_PASS_X_M
                else:
                    passing = abs(match["angle"]) >= CONE_PASSING_ANGLE_DEG

                if passing:
                    state = f"PASSING_{side}"
                else:
                    state = f"DETECTED_{side}"

                self.last_state = state

                return {
                    "state": state,
                    "cluster": dict(match),
                    "wall_like_ignored": self._wall_like_count(clusters),
                }

            self.lost_streak += 1

            if self.lost_streak < CONE_LOST_FRAMES:
                state = "TRACK_LOST_TEMP"
                self.last_state = state

                return {
                    "state": state,
                    "cluster": dict(self.locked),
                    "wall_like_ignored": self._wall_like_count(clusters),
                }

            # PASSING 상태까지 갔던 물체가 사라졌다면 통과 완료로 기록.
            if self.corner_hint_last:
                was_passing = (
                    self.last_state.startswith("PASSING_")
                    or float(self.locked["cx"]) <= CONE_CORNER_PASS_X_M
                )
            else:
                was_passing = (
                    self.last_state.startswith("PASSING_")
                    or abs(self.locked["angle"]) >= CONE_PASSING_ANGLE_DEG
                )

            self.locked = None
            self.lost_streak = 0
            self.candidate = None
            self.candidate_streak = 0

            if was_passing:
                self.passed_latch = 1
                self.last_state = "PASSED"
                return {
                    "state": "PASSED",
                    "cluster": None,
                    "wall_like_ignored": self._wall_like_count(clusters),
                }

        # ----------------------------------------------------
        # SEARCH / CANDIDATE
        # ----------------------------------------------------
        candidate = self._best_search_candidate(
            clusters,
            corner_hint=self.corner_hint_last,
        )

        if candidate is None:
            self.candidate = None
            self.candidate_streak = 0

            state = (
                "PASSED"
                if self.passed_latch > 0
                else "NONE"
            )

            self.last_state = state

            return {
                "state": state,
                "cluster": None,
                "wall_like_ignored": self._wall_like_count(clusters),
            }

        if self.candidate is None:
            self.candidate = candidate
            self.candidate_streak = 1

        else:
            center_shift = _cluster_center_distance(
                self.candidate,
                candidate,
            )

            if center_shift <= CONE_TRACK_MAX_CENTER_SHIFT_M:
                self.candidate = candidate
                self.candidate_streak += 1
            else:
                self.candidate = candidate
                self.candidate_streak = 1

        confirm_needed = CONE_CONFIRM_FRAMES
        if (
            self.corner_hint_last
            and float(candidate["dmin"]) <= CONE_CORNER_URGENT_RANGE_M
        ):
            confirm_needed = 1

        if self.candidate_streak >= confirm_needed:
            self.locked = self.candidate
            self.lost_streak = 0

            side = _cone_side(
                self.locked["angle"]
            )

            state = f"DETECTED_{side}"
            self.last_state = state

            return {
                "state": state,
                "cluster": dict(self.locked),
                "wall_like_ignored": self._wall_like_count(clusters),
            }

        side = _cone_side(candidate["angle"])
        state = f"CANDIDATE_{side}"
        self.last_state = state

        return {
            "state": state,
            "cluster": dict(candidate),
            "wall_like_ignored": self._wall_like_count(clusters),
        }

    @staticmethod
    def _wall_like_count(clusters):
        """
        참고용 숫자만 출력.
        벽을 '회피'하지 않으며 detector 제어에도 사용하지 않음.
        """
        return sum(
            1
            for c in clusters
            if (
                c["width"] >= 0.50
                or c["angle_span"] >= 30.0
            )
        )


def print_cone_result(result):
    state = result["state"]
    c = result["cluster"]
    wall_count = result["wall_like_ignored"]

    if c is None:
        print(
            f"[CONE] {state:<16} "
            f"wall_like_ignored={wall_count}"
        )
        return

    print(
        f"[CONE] {state:<16} "
        f"d={c['dmin']:.2f}m "
        f"a={c['angle']:+.1f}deg "
        f"w={c['width']:.2f}m "
        f"n={c['n']} "
        f"asp={c['angle_span']:.1f}deg "
        f"x={c['cx']:.2f}m "
        f"y={c['cy']:+.2f}m "
        f"wall_like_ignored={wall_count}"
    )


# ============================================================
# 6. IMAGE PROCESSING
# ============================================================

def roi_points(w, h):
    pts = ROI_NORM.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.float32)


def build_bev_matrices(w, h):
    src = roi_points(w, h)

    margin = 0.15 * w

    dst = np.float32([
        [margin, 0],
        [w - margin, 0],
        [w - margin, h - 1],
        [margin, h - 1],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return src, M, Minv


def make_white_mask(bev_bgr):
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, WHITE_V_MIN], dtype=np.uint8)
    upper = np.array([179, WHITE_S_MAX, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones(
        (MORPH_KERNEL, MORPH_KERNEL),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _search_base(histogram, x0, x1):
    x0 = max(0, int(x0))
    x1 = min(len(histogram), int(x1))

    if x1 <= x0:
        return None

    section = histogram[x0:x1]

    if section.size == 0 or np.max(section) <= 0:
        return None

    return int(x0 + np.argmax(section))


def sliding_window_fit(binary, side):
    """
    Fit x = f(y) for one boundary.
    Returns: (polyfit coefficients, number_of_pixels)
    """
    h, w = binary.shape

    histogram = np.sum(
        binary[int(h * 0.45):, :] > 0,
        axis=0
    )

    if side == "left":
        a, b = LEFT_SEARCH
    else:
        a, b = RIGHT_SEARCH

    base = _search_base(
        histogram,
        a * w,
        b * w
    )

    if base is None:
        return None, 0

    nonzero_y, nonzero_x = binary.nonzero()

    window_h = max(1, h // NWINDOWS)
    margin = max(10, int(w * WINDOW_MARGIN_RATIO))
    x_current = base

    lane_inds = []

    for win in range(NWINDOWS):
        y_low = h - (win + 1) * window_h
        y_high = h - win * window_h

        x_low = x_current - margin
        x_high = x_current + margin

        good = (
            (nonzero_y >= y_low) &
            (nonzero_y < y_high) &
            (nonzero_x >= x_low) &
            (nonzero_x < x_high)
        )

        inds = np.where(good)[0]

        if inds.size:
            lane_inds.append(inds)

        if inds.size >= MINPIX:
            x_current = int(np.mean(nonzero_x[inds]))

    if not lane_inds:
        return None, 0

    lane_inds = np.concatenate(lane_inds)

    xs = nonzero_x[lane_inds]
    ys = nonzero_y[lane_inds]

    if len(xs) < MIN_FIT_PIXELS:
        return None, len(xs)

    fit = np.polyfit(ys, xs, 2)
    return fit, len(xs)


def fit_x(fit, y):
    if fit is None:
        return None

    return float(
        fit[0] * y * y +
        fit[1] * y +
        fit[2]
    )


class LaneTracker:
    def __init__(self):
        self.lane_width_px = None

    def detect(self, mask):
        h, w = mask.shape

        y_near = int(h * NEAR_Y_RATIO)
        y_far  = int(h * FAR_Y_RATIO)

        left_fit, left_count = sliding_window_fit(mask, "left")
        right_fit, right_count = sliding_window_fit(mask, "right")

        left_near  = fit_x(left_fit, y_near)
        right_near = fit_x(right_fit, y_near)

        left_far  = fit_x(left_fit, y_far)
        right_far = fit_x(right_fit, y_far)

        both = (
            left_near is not None and
            right_near is not None and
            left_far is not None and
            right_far is not None
        )

        if both:
            width_near = right_near - left_near
            width_far  = right_far - left_far

            min_width = w * LANE_WIDTH_MIN_RATIO
            max_width = w * LANE_WIDTH_MAX_RATIO

            geometry_ok = (
                min_width <= width_near <= max_width and
                min_width <= width_far <= max_width and
                left_near < right_near and
                left_far < right_far
            )

            if not geometry_ok:
                both = False

        if both:
            observed_width = 0.5 * (
                (right_near - left_near) +
                (right_far - left_far)
            )

            if self.lane_width_px is None:
                self.lane_width_px = observed_width
            else:
                self.lane_width_px = (
                    LANE_WIDTH_ALPHA * observed_width +
                    (1.0 - LANE_WIDTH_ALPHA) * self.lane_width_px
                )

            center_near = 0.5 * (left_near + right_near)
            center_far  = 0.5 * (left_far + right_far)

            mode = "BOTH"
            confidence = 1.0

        elif self.lane_width_px is not None:
            if left_near is not None and left_far is not None:
                center_near = left_near + self.lane_width_px / 2.0
                center_far  = left_far + self.lane_width_px / 2.0

                mode = "LEFT_ONLY"
                confidence = 0.60

            elif right_near is not None and right_far is not None:
                center_near = right_near - self.lane_width_px / 2.0
                center_far  = right_far - self.lane_width_px / 2.0

                mode = "RIGHT_ONLY"
                confidence = 0.60

            else:
                center_near = None
                center_far = None

                mode = "LOST"
                confidence = 0.0

        else:
            center_near = None
            center_far = None

            mode = "LOST"
            confidence = 0.0

        if center_near is not None:
            if not (-0.15 * w <= center_near <= 1.15 * w):
                center_near = None
                center_far = None
                mode = "LOST"
                confidence = 0.0

        return {
            "left_fit": left_fit,
            "right_fit": right_fit,
            "left_count": left_count,
            "right_count": right_count,
            "left_near": left_near,
            "right_near": right_near,
            "left_far": left_far,
            "right_far": right_far,
            "center_near": center_near,
            "center_far": center_far,
            "confidence": confidence,
            "mode": mode,
            "lane_width_px": self.lane_width_px,
            "y_near": y_near,
            "y_far": y_far,
        }


# ============================================================
# 6-B. SHARP-CORNER PATH EXTRACTION
# ============================================================

def make_yellow_corner_mask(bev_bgr):
    """
    Extract yellow/orange center-line pieces, but only where they are
    on/near dark asphalt. This is intentionally NOT the normal lane detector.
    """
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(
        hsv,
        np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN], dtype=np.uint8),
        np.array([YELLOW_H_MAX, 255, 255], dtype=np.uint8),
    )

    # Asphalt context. Road is much darker than the surrounding grass.
    v = hsv[:, :, 2]
    asphalt = cv2.inRange(v, 0, ASPHALT_V_MAX)

    k = np.ones((ASPHALT_DILATE, ASPHALT_DILATE), np.uint8)
    asphalt_near = cv2.dilate(asphalt, k, iterations=1)

    yellow = cv2.bitwise_and(yellow, asphalt_near)

    k3 = np.ones((3, 3), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k3)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k3)

    return yellow


def yellow_centroids(yellow_mask):
    contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = yellow_mask.shape
    pts = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < YELLOW_MIN_AREA or area > YELLOW_MAX_AREA:
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue

        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

        # Ignore tiny edge artifacts.
        if not (0.03 * w <= cx <= 0.97 * w):
            continue
        if not (0.05 * h <= cy <= 0.98 * h):
            continue

        pts.append((cx, cy, area))

    return pts


def build_corner_path(points, w, h):
    """
    Chain yellow dash centroids in 2-D.
    Unlike x=f(y), this path can turn sideways through a 90-degree corner.
    """
    if not points:
        return []

    start = np.array([w / 2.0, h - 8.0], dtype=np.float32)
    unused = list(points)

    # First point: closest useful center-line piece to the vehicle.
    dists = [
        np.linalg.norm(np.array([p[0], p[1]], dtype=np.float32) - start)
        for p in unused
    ]

    first_i = int(np.argmin(dists))
    if dists[first_i] > CORNER_FIRST_LINK_PX:
        return []

    first = unused.pop(first_i)
    path = [(first[0], first[1])]
    current = np.array([first[0], first[1]], dtype=np.float32)

    for _ in range(14):
        if not unused:
            break

        best_i = None
        best_cost = None

        for i, p in enumerate(unused):
            q = np.array([p[0], p[1]], dtype=np.float32)
            delta = q - current
            dist = float(np.linalg.norm(delta))

            if dist > CORNER_MAX_LINK_PX:
                continue

            # In normal perspective, "forward" is decreasing y.
            # At a 90-degree bend, y may stay similar while x changes,
            # so allow modest sideways/backward movement but penalize it.
            backward = max(
                0.0,
                float(p[1] - current[1] - CORNER_BACKWARD_ALLOW_PX),
            )
            cost = dist + 3.0 * backward

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i = i

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
    prev = start
    travelled = 0.0
    target = np.array(path[-1], dtype=np.float32)

    for p in path:
        q = np.array(p, dtype=np.float32)
        travelled += float(np.linalg.norm(q - prev))
        target = q

        if travelled >= CORNER_LOOKAHEAD_PX:
            break

        prev = q

    return (float(target[0]), float(target[1]))


def steering_from_corner_target(target, w, h):
    if target is None:
        return None

    vehicle_x = w / 2.0
    vehicle_y = h - 8.0

    dx = float(target[0] - vehicle_x)
    forward = float(vehicle_y - target[1])

    # If the target has moved nearly sideways in a square corner,
    # retain a small positive forward term so atan2 remains well-defined.
    forward = max(12.0, forward)

    target_angle_deg = math.degrees(math.atan2(dx, forward))

    # Positive PhysiCar steering = LEFT.
    steering = -CORNER_STEER_GAIN * target_angle_deg

    return float(np.clip(
        steering,
        -MAX_STEERING_DEG,
        MAX_STEERING_DEG,
    ))


def draw_corner_debug(bev_vis, yellow_mask, path, target):
    # Yellow centerline candidates in magenta.
    contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(bev_vis, contours, -1, (255, 0, 255), 1)

    if path:
        pts = np.array(
            [(int(x), int(y)) for x, y in path],
            dtype=np.int32
        )

        if len(pts) >= 2:
            cv2.polylines(
                bev_vis,
                [pts],
                False,
                (255, 0, 255),
                3,
                cv2.LINE_AA,
            )

        for x, y in pts:
            cv2.circle(bev_vis, (int(x), int(y)), 4, (255, 0, 255), -1)

    if target is not None:
        cv2.circle(
            bev_vis,
            (int(target[0]), int(target[1])),
            10,
            (0, 0, 255),
            -1,
        )

# ============================================================
# 7. STEERING
# ============================================================

def steering_from_lane(det, w):
    center_near = det["center_near"]
    center_far = det["center_far"]

    if center_near is None or center_far is None:
        return None, None, None

    lateral_error = (
        center_near - w / 2.0
    ) / (w / 2.0)

    heading_error = (
        center_far - center_near
    ) / (w / 2.0)

    # positive steering = left
    steering = -(
        K_LATERAL * lateral_error +
        K_HEADING * heading_error
    )

    steering = float(np.clip(
        steering,
        -MAX_STEERING_DEG,
        MAX_STEERING_DEG,
    ))

    return steering, lateral_error, heading_error



def steering_from_lane_with_offsets(
    det,
    w,
    near_offset_px,
    far_offset_px,
):
    """
    Existing lane steering equation with only the desired lane-center
    targets shifted sideways.

    Image/BEV x:
      +offset = target moves RIGHT
      -offset = target moves LEFT

    PhysiCar steering:
      +steering = LEFT
      -steering = RIGHT
    """
    center_near = det["center_near"]
    center_far = det["center_far"]

    if center_near is None or center_far is None:
        return None, None, None

    target_near = center_near + float(near_offset_px)
    target_far = center_far + float(far_offset_px)

    lateral_error = (
        target_near - w / 2.0
    ) / (w / 2.0)

    heading_error = (
        target_far - target_near
    ) / (w / 2.0)

    steering = -(
        K_LATERAL * lateral_error
        + K_HEADING * heading_error
    )

    steering = float(np.clip(
        steering,
        -MAX_STEERING_DEG,
        MAX_STEERING_DEG,
    ))

    return steering, lateral_error, heading_error


def make_drivable_road_mask(bev_bgr):
    """
    Build a conservative drivable-asphalt mask for the corner guard.

    The existing corner detector already assumes asphalt is darker than the
    surroundings (ASPHALT_V_MAX). We reuse that verified cue, close over lane
    paint gaps, and retain only the connected component that reaches the
    vehicle-side bottom-center region.
    """
    if bev_bgr is None or bev_bgr.size == 0:
        return None

    h, w = bev_bgr.shape[:2]
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    road = cv2.inRange(v, 0, ASPHALT_V_MAX)

    close_k = max(3, int(ROAD_GUARD_CLOSE_KERNEL))
    if close_k % 2 == 0:
        close_k += 1
    open_k = max(3, int(ROAD_GUARD_OPEN_KERNEL))
    if open_k % 2 == 0:
        open_k += 1

    road = cv2.morphologyEx(
        road,
        cv2.MORPH_CLOSE,
        np.ones((close_k, close_k), np.uint8),
    )
    road = cv2.morphologyEx(
        road,
        cv2.MORPH_OPEN,
        np.ones((open_k, open_k), np.uint8),
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (road > 0).astype(np.uint8),
        connectivity=8,
    )

    if n_labels <= 1:
        return None

    # Prefer the component actually connected to the vehicle's current road.
    y1 = int(h * 0.76)
    y2 = h
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    seed_labels = labels[y1:y2, x1:x2].reshape(-1)
    seed_labels = seed_labels[seed_labels > 0]

    chosen = None
    if seed_labels.size:
        counts = np.bincount(seed_labels)
        chosen = int(np.argmax(counts))

    if chosen is None or chosen <= 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return None
        chosen = 1 + int(np.argmax(areas))

    area = float(stats[chosen, cv2.CC_STAT_AREA])
    if area < ROAD_GUARD_MIN_COMPONENT_RATIO * float(h * w):
        return None

    component = np.zeros((h, w), dtype=np.uint8)
    component[labels == chosen] = 255
    return component


def _corner_target_index_and_tangent(path, w, h):
    """Return the normal corner target plus a unit local path tangent."""
    if not path:
        return None, None, None

    start = np.array([w / 2.0, h - 8.0], dtype=np.float32)
    prev = start
    travelled = 0.0
    target_i = 0

    for i, p in enumerate(path):
        q = np.array(p, dtype=np.float32)
        travelled += float(np.linalg.norm(q - prev))
        target_i = i
        if travelled >= CORNER_LOOKAHEAD_PX:
            break
        prev = q

    target = np.array(path[target_i], dtype=np.float32)

    if len(path) == 1:
        tangent_vec = target - start
    else:
        p_prev = (
            start
            if target_i == 0
            else np.array(path[target_i - 1], dtype=np.float32)
        )
        p_next = (
            np.array(path[target_i + 1], dtype=np.float32)
            if target_i + 1 < len(path)
            else target
        )
        tangent_vec = p_next - p_prev

    norm = float(np.linalg.norm(tangent_vec))
    if norm < 1e-6:
        return (float(target[0]), float(target[1])), target_i, None

    tangent = tangent_vec / norm
    return (float(target[0]), float(target[1])), target_i, tangent


def shifted_corner_target_with_road_guard(
    path,
    w,
    h,
    requested_shift_px,
    road_mask,
    lane_width_px,
):
    """
    Shift the sharp-corner reference target along the LOCAL path normal.

    requested_shift_px:
      positive -> path-relative RIGHT
      negative -> path-relative LEFT

    Returns (target, accepted_scale, guard_state).
    accepted_scale=1 means the full requested avoidance offset was safe.
    accepted_scale=0 means the unshifted corner path was kept.
    """
    base_target, _, tangent = _corner_target_index_and_tangent(path, w, h)
    if base_target is None:
        return None, 0.0, "NO_PATH"

    if tangent is None:
        return base_target, 0.0, "NO_TANGENT"

    # Image coordinates: x right, y down. Path points run from vehicle toward
    # the road ahead. For straight-ahead tangent (0,-1), (-ty, tx)=(+1,0),
    # which is exactly image-right.
    tx, ty = float(tangent[0]), float(tangent[1])
    right_normal = np.array([-ty, tx], dtype=np.float32)

    shift = float(requested_shift_px)
    base = np.array(base_target, dtype=np.float32)

    if abs(shift) < 1e-3:
        return base_target, 1.0, "ZERO_REQUEST"

    if not ROAD_GUARD_ENABLED:
        target = base + right_normal * shift
        return (float(target[0]), float(target[1])), 1.0, "GUARD_DISABLED"

    # Safety-first fallback: when the road mask itself is unavailable in a
    # sharp corner, do NOT apply an unverified lateral shift. Keep following
    # the original corner path instead.
    if road_mask is None:
        return base_target, 0.0, "NO_ROAD_MASK"

    # Distance transform gives pixel clearance to the nearest non-road pixel.
    road_binary = (road_mask > 0).astype(np.uint8)
    clearance = cv2.distanceTransform(road_binary, cv2.DIST_L2, 5)

    if lane_width_px is not None and np.isfinite(lane_width_px):
        margin = max(
            ROAD_GUARD_MIN_MARGIN_PX,
            ROAD_GUARD_MARGIN_LANE_RATIO * float(lane_width_px),
        )
    else:
        margin = ROAD_GUARD_MIN_MARGIN_PX

    def safe(point):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (0 <= x < w and 0 <= y < h):
            return False
        return float(clearance[y, x]) >= margin

    # If even the original corner target is not supported by the asphalt mask,
    # do not trust this mask for steering changes. Preserve the original path.
    if not safe(base):
        return base_target, 0.0, "BASE_NOT_SAFE"

    steps = max(2, int(ROAD_GUARD_SHIFT_STEPS))
    for scale in np.linspace(1.0, 0.0, steps):
        candidate = base + right_normal * (shift * float(scale))
        if safe(candidate):
            state = "FULL" if scale >= 0.999 else "CLAMPED"
            return (
                (float(candidate[0]), float(candidate[1])),
                float(scale),
                state,
            )

    return base_target, 0.0, "ZERO_SHIFT"



def _raw_cluster_lateral_clearance(cluster):
    """Nearest cluster edge distance from vehicle centerline [m]."""
    y_min = float(cluster.get("y_min", cluster.get("cy", 999.0)))
    y_max = float(cluster.get("y_max", cluster.get("cy", 999.0)))

    if y_min > 0.0:
        return y_min
    if y_max < 0.0:
        return -y_max
    return 0.0


def find_raw_collision_cluster(clusters):
    """Return closest compact cluster that is still near the vehicle corridor."""
    hits = []
    for c in clusters:
        if (
            c.get("n", 0) >= RAW_SHIELD_MIN_POINTS
            and float(c.get("width", 999.0)) <= RAW_SHIELD_MAX_WIDTH_M
            and 0.0 < float(c.get("cx", -1.0)) <= RAW_SHIELD_BRAKE_X_M
            and _raw_cluster_lateral_clearance(c) <= RAW_SHIELD_SOFT_HALF_WIDTH_M
        ):
            hits.append(c)

    if not hits:
        return None

    return min(hits, key=lambda c: float(c.get("cx", 999.0)))


def raw_collision_shield_speed_cap(cluster, allow_escape_crawl=False):
    """
    Last-resort longitudinal safety cap.

    V13 fix:
    - STOP only when the *actual cluster edge* overlaps the hard vehicle corridor.
    - A cone that has already reached vehicle-side x and is outside the hard
      corridor is allowed to pass instead of pinning the car beside the cone.
    - Soft-corridor objects ahead still get a crawl cap.
    """
    if cluster is None:
        return None

    cx = float(cluster.get("cx", 999.0))
    lateral_clearance = _raw_cluster_lateral_clearance(cluster)

    # The object is already alongside the vehicle. If its nearest edge is
    # outside the hard footprint corridor, continuing forward is safer than
    # stopping forever next to the cone.
    if (
        cx <= RAW_SHIELD_SIDE_PASS_X_M
        and lateral_clearance > RAW_SHIELD_HARD_HALF_WIDTH_M
    ):
        return None

    # Definite footprint overlap. If a validated avoidance path is already
    # steering away from this same-side obstacle, keep a tiny forward crawl so
    # steering can actually generate lateral displacement. Hard-stop only at
    # truly immediate distance.
    if (
        cx <= RAW_SHIELD_STOP_X_M
        and lateral_clearance <= RAW_SHIELD_HARD_HALF_WIDTH_M
    ):
        if allow_escape_crawl and cx > RAW_SHIELD_ABSOLUTE_STOP_X_M:
            return RAW_SHIELD_ESCAPE_CRAWL_SPEED
        return 0.0

    # Near the footprint but not definitely overlapping -> crawl, don't stop.
    if (
        cx <= RAW_SHIELD_BRAKE_X_M
        and lateral_clearance <= RAW_SHIELD_SOFT_HALF_WIDTH_M
    ):
        return RAW_SHIELD_CRAWL_SPEED

    return None


def corner_collision_shield_speed_cap(
    cone_result,
    corner_active,
    guard_state,
    guard_scale,
):
    """Return None, a crawl cap, or 0.0 for last-resort collision prevention."""
    if not corner_active or cone_result is None:
        return None

    cone = cone_result.get("cluster")
    if cone is None:
        return None

    cx = float(cone.get("cx", 999.0))
    cy = abs(float(cone.get("cy", 999.0)))

    if cx <= 0.0:
        return None

    guard_useful = (
        guard_state in ("FULL", "CLAMPED", "ZERO_REQUEST")
        and float(guard_scale) >= CORNER_GUARD_MIN_USEFUL_SCALE
    )

    if guard_useful:
        return None

    if (
        cx <= CORNER_EMERGENCY_STOP_X_M
        and cy <= CORNER_EMERGENCY_STOP_ABS_Y_M
    ):
        return 0.0

    if (
        cx <= CORNER_UNSAFE_BRAKE_X_M
        and cy <= CORNER_UNSAFE_BRAKE_ABS_Y_M
    ):
        return CORNER_UNSAFE_SPEED

    return None

class ConeAvoidanceFSM:
    """
    Geometry-driven cone avoidance.

    NORMAL
      -> SHIFT_OUT
      -> PASS
      -> SHIFT_IN
      -> RECOVER
      -> NORMAL

    Avoid direction is latched once at obstacle acquisition.
    """

    def __init__(self):
        self.state = "NORMAL"

        # +1: shift desired lane center to image RIGHT = vehicle avoids RIGHT
        # -1: shift desired lane center to image LEFT  = vehicle avoids LEFT
        self.offset_sign = 0.0
        self.avoid_direction = "NONE"
        self.direction_reason = "NONE"

        self.near_offset_ratio = 0.0
        self.far_offset_ratio = 0.0

        self.last_cone = None

        # 회피 시작 순간의 lane width를 고정해 offset 크기가
        # 회피 중 lane detector 흔들림에 따라 변하지 않게 한다.
        self.latched_lane_width_px = None

    @staticmethod
    def _is_detected(state):
        return state.startswith("DETECTED_")

    @staticmethod
    def _is_passing(state):
        return state.startswith("PASSING_")

    def _choose_direction(
        self,
        cone,
        route_mode_hint=None,
        route_steering_hint=None,
    ):
        """
        Provisional direction from obstacle geometry.

        IMPORTANT:
        - Do NOT force the route-turn direction in a corner.
        - Cone LEFT  -> provisional avoid RIGHT
        - Cone RIGHT -> provisional avoid LEFT
        - The current-frame corner dual-path planner may override this only
          when the preferred side is not drivable and the other side is.
        """
        cy = float(cone.get("cy", 0.0))

        if cy > AVOID_CENTER_DEADBAND_M:
            self.avoid_direction = "RIGHT"
            self.offset_sign = +1.0
            self.direction_reason = "CONE_LEFT"

        elif cy < -AVOID_CENTER_DEADBAND_M:
            self.avoid_direction = "LEFT"
            self.offset_sign = -1.0
            self.direction_reason = "CONE_RIGHT"

        else:
            angle = float(cone.get("angle", 0.0))

            if angle > 0.0:
                self.avoid_direction = "RIGHT"
                self.offset_sign = +1.0
                self.direction_reason = "CENTER_ANGLE_LEFT"
            else:
                self.avoid_direction = "LEFT"
                self.offset_sign = -1.0
                self.direction_reason = "CENTER_ANGLE_RIGHT"

    def _latch_lane_width(self, current_lane_width_px):
        if (
            current_lane_width_px is not None
            and np.isfinite(current_lane_width_px)
            and float(current_lane_width_px) > 50.0
        ):
            self.latched_lane_width_px = float(current_lane_width_px)

    def update_observation(
        self,
        cone_result,
        current_lane_width_px=None,
        route_mode_hint=None,
        route_steering_hint=None,
    ):
        state = cone_result["state"]
        cone = cone_result["cluster"]

        if cone is not None:
            self.last_cone = dict(cone)

        # ----------------------------------------------------
        # V9.1: 복귀 중 새 라바콘이 확인되면 복귀를 중단하고
        # 즉시 새 회피를 시작한다.
        # ----------------------------------------------------
        if (
            self.state in ("SHIFT_IN", "RECOVER")
            and AVOIDANCE_ENABLED
            and self._is_detected(state)
            and cone is not None
            and cone["x_max"] > 0.0
        ):
            self._choose_direction(
                cone,
                route_mode_hint,
                route_steering_hint,
            )
            self._latch_lane_width(current_lane_width_px)
            self.state = "SHIFT_OUT"
            return

        if self.state == "NORMAL":
            if (
                AVOIDANCE_ENABLED
                and self._is_detected(state)
                and cone is not None
                and cone["x_max"] > 0.0
            ):
                self._choose_direction(
                    cone,
                    route_mode_hint,
                    route_steering_hint,
                )
                self._latch_lane_width(current_lane_width_px)
                self.state = "SHIFT_OUT"

        elif self.state == "SHIFT_OUT":
            if state == "PASSED":
                self.state = "SHIFT_IN"

            elif cone is not None:
                # Once enough lateral separation is obtained, hold the
                # offset and pass the cone.
                if (
                    abs(float(cone["cy"])) >= AVOID_SAFE_LATERAL_M
                    or self._is_passing(state)
                    or float(cone["cx"]) <= AVOID_PASS_X_M
                ):
                    self.state = "PASS"

        elif self.state == "PASS":
            # V13: if a clearly-ahead cone appears while PASS is still latched,
            # treat it as a new/reacquired obstacle. Carrying the previous PASS
            # direction into the next cone is what caused corner path drift.
            if (
                self._is_detected(state)
                and cone is not None
                and float(cone.get("cx", -1.0)) >= CORNER_REACQUIRE_X_M
            ):
                self._choose_direction(
                    cone,
                    route_mode_hint,
                    route_steering_hint,
                )
                self._latch_lane_width(current_lane_width_px)
                self.state = "SHIFT_OUT"
                return

            if state == "PASSED":
                self.state = "SHIFT_IN"

            elif cone is not None and float(cone["cx"]) < -0.03:
                self.state = "SHIFT_IN"

        # SHIFT_IN / RECOVER transitions are completed from offset geometry
        # in step_offsets(), not from timers.

    def _target_offsets(self):
        s = self.offset_sign

        if self.state == "NORMAL":
            return 0.0, 0.0

        if self.state == "SHIFT_OUT":
            return (
                s * AVOID_SHIFT_NEAR_RATIO,
                s * AVOID_FULL_OFFSET_RATIO,
            )

        if self.state == "PASS":
            return (
                s * AVOID_FULL_OFFSET_RATIO,
                s * AVOID_FULL_OFFSET_RATIO,
            )

        if self.state == "SHIFT_IN":
            # Far target returns first. This points the vehicle back toward
            # the original lane center while the near target stays offset.
            return (
                s * AVOID_FULL_OFFSET_RATIO,
                0.0,
            )

        if self.state == "RECOVER":
            return 0.0, 0.0

        return 0.0, 0.0

    def step_offsets(self):
        target_near, target_far = self._target_offsets()

        self.near_offset_ratio = (
            AVOID_OFFSET_ALPHA * target_near
            + (1.0 - AVOID_OFFSET_ALPHA) * self.near_offset_ratio
        )

        self.far_offset_ratio = (
            AVOID_OFFSET_ALPHA * target_far
            + (1.0 - AVOID_OFFSET_ALPHA) * self.far_offset_ratio
        )

        if self.state == "SHIFT_IN":
            if abs(self.far_offset_ratio) <= 0.035:
                self.state = "RECOVER"

        elif self.state == "RECOVER":
            if (
                abs(self.near_offset_ratio) <= AVOID_OFFSET_DONE_RATIO
                and abs(self.far_offset_ratio) <= AVOID_OFFSET_DONE_RATIO
            ):
                self.state = "NORMAL"
                self.offset_sign = 0.0
                self.avoid_direction = "NONE"
                self.direction_reason = "NONE"
                self.near_offset_ratio = 0.0
                self.far_offset_ratio = 0.0
                self.last_cone = None
                self.latched_lane_width_px = None

        return self.near_offset_ratio, self.far_offset_ratio

    def speed_cap(self):
        if self.state == "SHIFT_OUT":
            cap = AVOID_SHIFT_SPEED

        elif self.state == "PASS":
            cap = AVOID_PASS_SPEED

        elif self.state == "SHIFT_IN":
            cap = AVOID_RETURN_SPEED

        elif self.state == "RECOVER":
            cap = AVOID_RECOVER_SPEED

        else:
            return None

        cone = self.last_cone

        if cone is not None:
            cx = float(cone["cx"])
            cy = abs(float(cone["cy"]))

            if 0.0 < cx <= AVOID_LATE_X_M:
                cap = min(cap, AVOID_LATE_SPEED)

            if (
                0.0 < cx <= AVOID_EMERGENCY_X_M
                and cy <= AVOID_EMERGENCY_LATERAL_M
            ):
                cap = min(cap, AVOID_EMERGENCY_SPEED)

        return cap


def lane_center_at_y(det, y):
    """Return lane center x at BEV y using the fitted boundaries."""
    left = fit_x(det["left_fit"], y)
    right = fit_x(det["right_fit"], y)

    if left is not None and right is not None:
        return 0.5 * (left + right)

    # One-line fallback using remembered lane width.
    width = det.get("lane_width_px")

    if width is not None:
        if left is not None:
            return left + width / 2.0
        if right is not None:
            return right - width / 2.0

    return None


def preview_lane_geometry(det, h, w, current_speed):
    """
    Sample the lane center from NEAR -> FAR in the correct BEV order.

    Important:
      In BEV, a larger y means closer to the vehicle.
      Therefore the sample order must be:
          near (large y) -> far (small y)

    The previous version used np.unique(), which sorted the y-ratios in
    ascending order. That reversed the near/far interpretation and could
    make a perfectly straight lane produce preview_curvature ~= 1.0.
    """
    speed_ratio = np.clip(
        current_speed / max(SPEED_MAX, 1e-6),
        0.0,
        1.0,
    )

    # Faster vehicle -> look farther ahead.
    # Keep the far point genuinely ahead of the near point.
    far_ratio = 0.55 - 0.10 * speed_ratio
    far_ratio = float(np.clip(
        far_ratio,
        PREVIEW_MIN_Y_RATIO,
        0.60,
    ))

    # NEVER sort these with np.unique().
    # They are deliberately ordered near -> far.
    ratios = np.linspace(
        0.82,
        far_ratio,
        5,
        dtype=np.float32,
    )

    centers = []

    for r in ratios:
        y = int(h * float(r))
        x = lane_center_at_y(det, y)

        if x is not None and -0.15 * w <= x <= 1.15 * w:
            centers.append((y, x))

    if len(centers) < 3:
        return None, 0.0, centers

    # centers[0] = near, centers[-1] = far.
    y_near, x_near = centers[0]
    y_far, x_far = centers[-1]

    dy_total = max(
        1.0,
        float(y_near - y_far),
    )

    dx_total = float(x_far - x_near)

    # Future path direction.
    preview_heading = dx_total / (w / 2.0)

    # Estimate local path slopes.
    slopes = []

    for i in range(len(centers) - 1):
        y0, x0 = centers[i]
        y1, x1 = centers[i + 1]

        dy_seg = max(
            1.0,
            float(y0 - y1),
        )

        dx_seg = float(x1 - x0)

        slopes.append(dx_seg / dy_seg)

    # Straight lane -> all slopes approximately equal.
    # Curved lane -> slope changes along the preview.
    if len(slopes) >= 2:
        slope_changes = np.abs(
            np.diff(np.asarray(slopes, dtype=np.float32))
        )

        # Keep this deliberately conservative.
        curvature_from_slope = np.clip(
            float(np.mean(slope_changes)) * 2.0,
            0.0,
            1.0,
        )
    else:
        curvature_from_slope = 0.0

    # Total heading change is a secondary signal.
    heading_level = np.clip(
        abs(preview_heading) / 0.30,
        0.0,
        1.0,
    )

    curvature_level = max(
        curvature_from_slope,
        heading_level * 0.65,
    )

    return (
        float(preview_heading),
        float(curvature_level),
        centers,
    )


def choose_target_speed(
    det,
    steering,
    heading_error,
    preview_curvature=0.0,
):
    """
    Speed planner with a deadband.

    Important:
      Preview geometry affects SPEED only.
      Steering remains the proven lane/corner controller.
    """
    if steering is None or heading_error is None:
        return 0.0

    if det["mode"] == "LOST":
        return 0.0

    if det["mode"] in ("LEFT_ONLY", "RIGHT_ONLY"):
        return SPEED_ONE_LINE

    s = abs(float(steering))
    h = abs(float(heading_error))
    p = float(preview_curvature)

    # Current path geometry.
    current_curve = max(
        np.clip(h / 0.22, 0.0, 1.0),
        np.clip(s / 18.0, 0.0, 1.0),
    )

    # Preview is useful, but should not dominate normal driving.
    combined_curve = max(
        p * 0.85,
        current_curve * 0.55,
    )

    # Straight-road deadband.
    # Small fitting noise is treated as zero curvature.
    if combined_curve <= STRAIGHT_CURVE_LIMIT:
        return SPEED_MAX

    # Remove the deadband before converting curvature to speed.
    effective_curve = np.clip(
        (combined_curve - CURVE_DEADBAND) /
        max(1.0 - CURVE_DEADBAND, 1e-6),
        0.0,
        1.0,
    )

    speed = SPEED_MAX - (
        effective_curve * (SPEED_MAX - SPEED_MIN)
    )

    return float(np.clip(
        speed,
        SPEED_MIN,
        SPEED_MAX,
    ))


def ramp_speed(current, target):
    if target > current:
        return min(target, current + SPEED_RAMP_UP)
    return max(target, current - SPEED_RAMP_DOWN)

# ============================================================
# 8. DEBUG DRAWING
# ============================================================

def draw_fit(img, fit, color, thickness=3):
    if fit is None:
        return

    h, w = img.shape[:2]

    ys = np.linspace(0, h - 1, 80)
    pts = []

    for y in ys:
        x = fit_x(fit, y)

        if x is not None and -50 <= x <= w + 50:
            pts.append(
                (int(x), int(y))
            )

    if len(pts) >= 2:
        cv2.polylines(
            img,
            [np.array(pts, dtype=np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def make_debug_panel(
    camera_bgr,
    src_pts,
    bev,
    mask,
    det,
    steering,
    lateral_error,
    heading_error,
):
    h, w = camera_bgr.shape[:2]

    # ---------------------------------
    # Top-left: camera + ROI
    # ---------------------------------
    camera_vis = camera_bgr.copy()

    cv2.polylines(
        camera_vis,
        [src_pts.astype(np.int32)],
        True,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        camera_vis,
        "CAMERA + ROI",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # ---------------------------------
    # Top-right: BEV
    # ---------------------------------
    bev_vis = bev.copy()

    # v5 sharp-corner debug:
    # magenta = filtered yellow path, red = current corner target
    corner_yellow_mask = det.get("corner_yellow_mask")
    corner_path = det.get("corner_path", [])
    corner_target = det.get("corner_target")

    if corner_yellow_mask is not None:
        draw_corner_debug(
            bev_vis,
            corner_yellow_mask,
            corner_path,
            corner_target,
        )

    # Cyan = final path-relative avoidance target after road-boundary guard.
    avoid_corner_target = det.get("avoid_corner_target")
    if avoid_corner_target is not None:
        cv2.circle(
            bev_vis,
            (int(avoid_corner_target[0]), int(avoid_corner_target[1])),
            9,
            (255, 255, 0),
            2,
        )

    draw_fit(
        bev_vis,
        det["left_fit"],
        (0, 255, 0),
    )

    draw_fit(
        bev_vis,
        det["right_fit"],
        (0, 255, 0),
    )

    y_near = det["y_near"]
    y_far  = det["y_far"]

    center_near = det["center_near"]
    center_far  = det["center_far"]

    cv2.line(
        bev_vis,
        (w // 2, 0),
        (w // 2, h - 1),
        (255, 255, 255),
        1,
    )

    if center_near is not None:
        cv2.circle(
            bev_vis,
            (int(center_near), y_near),
            8,
            (255, 255, 0),
            -1,
        )

    if center_far is not None:
        cv2.circle(
            bev_vis,
            (int(center_far), y_far),
            8,
            (255, 255, 0),
            -1,
        )

    if (
        center_near is not None and
        center_far is not None
    ):
        cv2.line(
            bev_vis,
            (int(center_near), y_near),
            (int(center_far), y_far),
            (255, 255, 0),
            3,
        )

    # Preview path: cyan circles connected through the future lane center.
    preview_centers = det.get("preview_centers", [])
    if preview_centers:
        pts = np.array(
            [(int(x), int(y)) for y, x in preview_centers],
            dtype=np.int32,
        )

        if len(pts) >= 2:
            cv2.polylines(
                bev_vis,
                [pts],
                False,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for x, y in pts:
            cv2.circle(
                bev_vis,
                (int(x), int(y)),
                5,
                (255, 255, 0),
                -1,
            )

    cv2.putText(
        bev_vis,
        "BEV  green=boundaries  cyan=center",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 0),
        2,
    )

    # ---------------------------------
    # Bottom-left: white mask
    # ---------------------------------
    mask_vis = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        mask_vis,
        "WHITE MASK",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # ---------------------------------
    # Bottom-right: information
    # ---------------------------------
    info = np.zeros_like(camera_bgr)

    lane_width = det["lane_width_px"]

    lines = [
        f"control: {det.get('control_mode', 'NORMAL')}",
        f"mode: {det['mode']}",
        f"confidence: {det['confidence']:.2f}",
        (
            f"lane_width_px: {lane_width:.1f}"
            if lane_width is not None
            else "lane_width_px: None"
        ),
        (
            f"steering: {steering:+.2f} deg"
            if steering is not None
            else "steering: None"
        ),
        (
            f"lateral: {lateral_error:+.3f}"
            if lateral_error is not None
            else "lateral: None"
        ),
        (
            f"heading: {heading_error:+.3f}"
            if heading_error is not None
            else "heading: None"
        ),
        (
            f"preview_heading: {det.get('preview_heading'):+.3f}"
            if det.get("preview_heading") is not None
            else "preview_heading: None"
        ),
        f"preview_curve: {det.get('preview_curvature', 0.0):.3f}",
        f"curve_deadband: {CURVE_DEADBAND:.2f}",
        f"preview_pts: {len(det.get('preview_centers', []))}",
        (
            "preview_x: " +
            ",".join(
                str(int(x))
                for _, x in det.get("preview_centers", [])
            )
            if det.get("preview_centers")
            else "preview_x: None"
        ),
        f"DRIVE: {'ON' if DRIVE_ENABLED else 'OFF - perception only'}",
        f"WHITE: S<={WHITE_S_MAX}, V>={WHITE_V_MIN}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            info,
            text,
            (18, 34 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    top = np.hstack([
        camera_vis,
        bev_vis,
    ])

    bottom = np.hstack([
        mask_vis,
        info,
    ])

    panel = np.vstack([
        top,
        bottom,
    ])

    return panel

# ============================================================
# 9. MY APP VIEWER — SAME PATTERN AS OFFICIAL line-tracing.ipynb
# ============================================================

# The official example does NOT use MJPEG.
# It serves:
#   /        -> assets/line-tracing/webui.html
#   /frame   -> latest single JPEG
#   /data    -> latest status JSON
# We reuse that exact viewer structure.

import os

WEBUI_DIR = "assets/line-tracing"
if not os.path.isdir(WEBUI_DIR):
    WEBUI_DIR = "examples/assets/line-tracing"

_frame = [b""]
_status = ["starting"]
_server = [None]


def stop_view():
    if _server[0]:
        try:
            _server[0].shutdown()
        except Exception:
            pass
        _server[0] = None


def _start(app):
    import logging
    from werkzeug.serving import make_server

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    stop_view()

    try:
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("", 5000))
        probe.close()

        server = make_server(
            "0.0.0.0",
            5000,
            app,
            threaded=True,
        )

    except (OSError, SystemExit):
        print(
            "[WEB] port 5000 is in use. "
            "Interrupt the other example/notebook cell, "
            "then run this file again."
        )
        return None

    _server[0] = server

    threading.Thread(
        target=server.serve_forever,
        daemon=True,
    ).start()

    print("[WEB] official MY APP viewer pattern started on port 5000")
    return server


def update_web(panel, status):
    ok, jpg = cv2.imencode(
        ".jpg",
        panel,
        [cv2.IMWRITE_JPEG_QUALITY, 82],
    )

    if ok:
        _frame[0] = jpg.tobytes()

    _status[0] = status


def serve():
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    webui_path = os.path.join(WEBUI_DIR, "webui.html")

    if not os.path.isfile(webui_path):
        raise FileNotFoundError(
            f"Official webui.html not found: {webui_path}"
        )

    # Use the exact page supplied by PhysiCar.
    page = open(
        webui_path,
        "r",
        encoding="utf-8",
    ).read()

    @app.get("/")
    def index():
        return page

    @app.get("/frame")
    def frame():
        # Avoid sending an empty image before the first camera frame arrives.
        if not _frame[0]:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "waiting for first frame...",
                (12, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            ok, jpg = cv2.imencode(".jpg", blank)
            body = jpg.tobytes() if ok else b""
        else:
            body = _frame[0]

        return Response(
            body,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/data")
    def data():
        return jsonify({
            "status": _status[0]
        })

    return _start(app)

# ============================================================
# 10. MAIN
# ============================================================

def main():
    tracker = LaneTracker()

    # LiDAR는 읽기/분석/로그만 수행합니다.
    # 차량 제어에는 연결하지 않습니다.
    start_lidar_cluster_logger()
    last_lidar_cluster_print = 0.0

    # READ-ONLY cone detector.
    # 어떤 결과가 나와도 차량 제어에는 사용하지 않습니다.
    cone_detector = ConeReadOnlyDetector()

    # V9 avoidance FSM.
    avoidance = ConeAvoidanceFSM()
    latest_cone_result = {
        "state": "NONE",
        "cluster": None,
        "wall_like_ignored": 0,
    }
    latest_raw_collision_cluster = None

    smoothed_steering = 0.0
    current_speed = 0.0
    last_print = 0.0

    corner_active = False
    corner_exit_count = 0
    corner_hold_count = 0
    last_corner_steering = 0.0

    # Previous-frame V8 route hint for corner-aware avoidance direction.
    last_v8_control_mode = "NORMAL"
    last_v8_control_steering = 0.0

    serve()

    # Explicitly clear any stale command from a previous run.
    if DRIVE_ENABLED:
        stop_vehicle()

    print("=" * 72)
    print("AMET LOG TEST V4 FRENET GUARD - V3 BASE")
    print(f"DRIVE_ENABLED = {DRIVE_ENABLED}")
    print(
        f"START GATE: SEARCH -> LOCK -> GREEN "
        f"{GREEN_CONFIRM_FRAMES} frames -> RETURN_CAMERA"
    )
    print("RED / UNKNOWN / NO LIGHT -> NEVER START")
    print("After GREEN latch, traffic detection is disabled permanently.")
    print("Lane controller = test_v7 behavior (unchanged)")
    print("LiDAR = V3 CONE FSM + corner-early detect + x-pass + footprint-aware shield")
    print("Walls are classified/logged but NEVER enter the avoidance FSM.")
    print("Ctrl+C to stop.")
    print("=" * 72)

    try:
        # Pre-race blocking gate. Vehicle remains stopped until confirmed GREEN
        # and camera is returned to the proven lane-driving pose.
        search_and_wait_for_green()

        # Clear control history before the first racing frame.
        current_speed = 0.0
        smoothed_steering = 0.0
        if "smoothed_target_speed" in locals():
            del smoothed_target_speed

        while True:
            loop_start = time.time()

            try:
                img = camera()

            except Exception as e:
                print(
                    f"[CAMERA] {type(e).__name__}: {e}"
                )

                if DRIVE_ENABLED:
                    stop_vehicle()

                time.sleep(0.2)
                continue

            h, w = img.shape[:2]

            src_pts, M, _ = build_bev_matrices(
                w,
                h,
            )

            bev = cv2.warpPerspective(
                img,
                M,
                (w, h),
            )

            mask = make_white_mask(bev)
            mask[:int(h * 0.18), :] = 0

            det = tracker.detect(mask)

            steering, lateral_error, heading_error = (
                steering_from_lane(
                    det,
                    w,
                )
            )

            # ------------------------------------------------
            # Pre-corner hint for LiDAR acquisition
            # ------------------------------------------------
            # This is deliberately available BEFORE the full yellow-corner
            # pipeline below, so a cone at the corner entrance is not searched
            # with the straight-road +/-35 deg gate.
            pre_corner_hint = (
                corner_active
                or det["mode"] in ("LEFT_ONLY", "RIGHT_ONLY", "LOST")
                or (
                    heading_error is not None
                    and abs(float(heading_error)) >= 0.10
                )
                or (
                    steering is not None
                    and abs(float(steering)) >= 7.0
                )
            )

            # ------------------------------------------------
            # V9 LiDAR cone observation update (5 Hz)
            # ------------------------------------------------
            cone_now = time.time()

            if (
                cone_now - last_lidar_cluster_print
                >= 1.0 / max(LIDAR_CLUSTER_HZ, 1e-6)
            ):
                clusters = get_lidar_clusters()
                latest_raw_collision_cluster = find_raw_collision_cluster(
                    clusters
                )
                latest_cone_result = cone_detector.update(
                    clusters,
                    corner_hint=pre_corner_hint,
                )
                avoidance.update_observation(
                    latest_cone_result,
                    det.get("lane_width_px"),
                    last_v8_control_mode,
                    last_v8_control_steering,
                )

                print_cone_result(latest_cone_result)
                last_lidar_cluster_print = cone_now

            # Offset target is smoothed at camera-loop rate.
            avoid_near_ratio, avoid_far_ratio = (
                avoidance.step_offsets()
            )

            # V9.1:
            # 회피 시작 시점의 lane width를 우선 사용한다.
            # 회피 도중 LEFT_ONLY/RIGHT_ONLY/CORNER로 lane width가 흔들려도
            # avoidance offset의 실제 pixel 크기는 일정하게 유지한다.
            if (
                avoidance.state != "NORMAL"
                and avoidance.latched_lane_width_px is not None
            ):
                lane_width_for_avoid = (
                    avoidance.latched_lane_width_px
                )
            else:
                lane_width_for_avoid = det.get("lane_width_px")

            if lane_width_for_avoid is not None:
                avoid_near_px = (
                    avoid_near_ratio * lane_width_for_avoid
                )
                avoid_far_px = (
                    avoid_far_ratio * lane_width_for_avoid
                )
            else:
                avoid_near_px = 0.0
                avoid_far_px = 0.0

            # The straight-road offset steering is intentionally deferred
            # until after current-frame corner detection. In a sharp corner we
            # will instead shift the CORNER PATH along its local normal.
            avoid_lane_steering = None

            # ------------------------------------------------
            # v5 CORNER detector/path
            # ------------------------------------------------
            yellow_corner_mask = make_yellow_corner_mask(bev)
            yellow_pts = yellow_centroids(yellow_corner_mask)
            corner_path = build_corner_path(yellow_pts, w, h)
            corner_target = choose_corner_target(corner_path, w, h)
            corner_steering = steering_from_corner_target(
                corner_target,
                w,
                h,
            )

            det["corner_yellow_mask"] = yellow_corner_mask
            det["corner_path"] = corner_path
            det["corner_target"] = corner_target

            normal_sharp = (
                heading_error is not None and
                abs(heading_error) >= CORNER_ENTER_HEADING
            )
            normal_big_steer = (
                steering is not None and
                abs(steering) >= CORNER_ENTER_STEER
            )
            weak_white = det["mode"] in (
                "LEFT_ONLY",
                "RIGHT_ONLY",
                "LOST",
            )

            # Enter corner mode only when geometry suggests a sharp bend
            # AND a plausible yellow path exists.
            if (
                not corner_active and
                corner_steering is not None and
                len(corner_path) >= 2 and
                (weak_white or normal_sharp or normal_big_steer)
            ):
                corner_active = True
                corner_exit_count = 0
                corner_hold_count = 0

            # Exit only after normal white-boundary tracking is stably back.
            if corner_active:
                white_recovered = (
                    det["mode"] == "BOTH" and
                    heading_error is not None and
                    abs(heading_error) < CORNER_EXIT_HEADING and
                    steering is not None and
                    abs(steering) < CORNER_EXIT_STEER
                )

                if white_recovered:
                    corner_exit_count += 1
                else:
                    corner_exit_count = 0

                if corner_exit_count >= CORNER_EXIT_BOTH_FRAMES:
                    corner_active = False
                    corner_exit_count = 0
                    corner_hold_count = 0

            # ------------------------------------------------
            # Curved-path avoidance target + drivable-road guard
            # ------------------------------------------------
            # Straight road: keep the original V9 offset-lane steering.
            # Sharp corner: shift the yellow reference path along its local
            # normal, NOT along global BEV x. This is the key geometry fix.
            avoid_lane_steering, _, _ = (
                steering_from_lane_with_offsets(
                    det,
                    w,
                    avoid_near_px,
                    avoid_far_px,
                )
            )

            road_guard_mask = None
            avoid_corner_target = None
            avoid_corner_steering = None
            avoid_corner_shift_scale = 0.0
            avoid_corner_guard_state = "INACTIVE"

            # Per-side debug values for V13.
            det["corner_left_scale"] = 0.0
            det["corner_right_scale"] = 0.0
            det["corner_plan_side"] = "NONE"
            det["corner_shift_ratio_cmd"] = 0.0

            if (
                ROAD_GUARD_ENABLED
                and corner_active
                and avoidance.state != "NORMAL"
                and len(corner_path) >= 1
            ):
                road_guard_mask = make_drivable_road_mask(bev)
                active_corner_cone = (
                    latest_cone_result.get("cluster")
                    if latest_cone_result is not None
                    else None
                )

                # SHIFT_OUT/PASS only modifies the corner path while there is
                # a CURRENT tracked cone. A stale PASS state by itself must not
                # pull a clean corner away from the proven V3 corner path.
                if (
                    avoidance.state in ("SHIFT_OUT", "PASS")
                    and active_corner_cone is not None
                    and float(active_corner_cone.get("cx", -1.0)) > -0.02
                    and lane_width_for_avoid is not None
                ):
                    cone_x = float(active_corner_cone.get("cx", 999.0))
                    proximity = float(np.clip(
                        (CORNER_AVOID_SHIFT_FAR_X_M - cone_x)
                        / max(
                            CORNER_AVOID_SHIFT_FAR_X_M
                            - CORNER_AVOID_SHIFT_NEAR_X_M,
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    ))
                    shift_ratio = (
                        CORNER_AVOID_SHIFT_FAR_RATIO
                        + proximity * (
                            CORNER_AVOID_SHIFT_NEAR_RATIO
                            - CORNER_AVOID_SHIFT_FAR_RATIO
                        )
                    )
                    shift_mag = (
                        shift_ratio * float(lane_width_for_avoid)
                    )
                    det["corner_shift_ratio_cmd"] = float(shift_ratio)

                    # Build BOTH candidates in the local path-normal frame.
                    # sign -1 = path-relative LEFT, +1 = path-relative RIGHT.
                    left_target, left_scale, left_state = (
                        shifted_corner_target_with_road_guard(
                            corner_path, w, h, -shift_mag, road_guard_mask,
                            lane_width_for_avoid,
                        )
                    )
                    right_target, right_scale, right_state = (
                        shifted_corner_target_with_road_guard(
                            corner_path, w, h, +shift_mag, road_guard_mask,
                            lane_width_for_avoid,
                        )
                    )

                    det["corner_left_scale"] = float(left_scale)
                    det["corner_right_scale"] = float(right_scale)

                    cy = float(active_corner_cone.get("cy", 0.0))
                    angle = float(active_corner_cone.get("angle", 0.0))
                    if cy > AVOID_CENTER_DEADBAND_M:
                        preferred_sign = +1.0   # cone LEFT -> avoid RIGHT
                        cone_reason = "CONE_LEFT"
                    elif cy < -AVOID_CENTER_DEADBAND_M:
                        preferred_sign = -1.0   # cone RIGHT -> avoid LEFT
                        cone_reason = "CONE_RIGHT"
                    elif angle > 0.0:
                        preferred_sign = +1.0
                        cone_reason = "CENTER_ANGLE_LEFT"
                    else:
                        preferred_sign = -1.0
                        cone_reason = "CENTER_ANGLE_RIGHT"

                    def _candidate_score(sign, scale):
                        score = float(scale)
                        if sign == preferred_sign:
                            score += CORNER_DUALPATH_PREFERENCE_BONUS
                        if sign == float(avoidance.offset_sign):
                            score += CORNER_DUALPATH_CONTINUITY_BONUS
                        if float(scale) < CORNER_DUALPATH_MIN_SCALE:
                            score -= 0.50
                        return score

                    left_score = _candidate_score(-1.0, left_scale)
                    right_score = _candidate_score(+1.0, right_scale)

                    if left_score >= right_score:
                        chosen_sign = -1.0
                        avoid_corner_target = left_target
                        avoid_corner_shift_scale = float(left_scale)
                        avoid_corner_guard_state = left_state
                        avoidance.avoid_direction = "LEFT"
                        det["corner_plan_side"] = "LEFT"
                    else:
                        chosen_sign = +1.0
                        avoid_corner_target = right_target
                        avoid_corner_shift_scale = float(right_scale)
                        avoid_corner_guard_state = right_state
                        avoidance.avoid_direction = "RIGHT"
                        det["corner_plan_side"] = "RIGHT"

                    avoidance.offset_sign = chosen_sign
                    avoidance.direction_reason = (
                        f"DUALPATH_{cone_reason}"
                    )

                    avoid_corner_steering = steering_from_corner_target(
                        avoid_corner_target, w, h
                    )

                else:
                    # No current obstacle, or already returning: keep the exact
                    # original corner target. This prevents stale avoidance state
                    # from causing path departure while simply cornering.
                    avoid_corner_target = corner_target
                    avoid_corner_shift_scale = 1.0
                    avoid_corner_guard_state = "BASE_PATH"
                    det["corner_plan_side"] = "BASE"
                    avoid_corner_steering = steering_from_corner_target(
                        avoid_corner_target, w, h
                    )

            det["road_guard_mask"] = road_guard_mask
            det["avoid_corner_target"] = avoid_corner_target
            det["avoid_corner_shift_scale"] = avoid_corner_shift_scale
            det["avoid_corner_guard_state"] = avoid_corner_guard_state

            control_steering = steering
            control_mode = "NORMAL"

            if corner_active:
                control_mode = "CORNER"

                if corner_steering is not None:
                    control_steering = corner_steering
                    last_corner_steering = corner_steering
                    corner_hold_count = 0

                elif corner_hold_count < CORNER_HOLD_FRAMES:
                    control_mode = "CORNER_HOLD"
                    control_steering = last_corner_steering
                    corner_hold_count += 1

                else:
                    # V9.4 SAFE FALLBACK
                    #
                    # Yellow corner path가 사라졌더라도 white lane이 살아 있으면
                    # 즉시 정지하지 않는다.
                    #
                    # 단, LEFT_ONLY/RIGHT_ONLY lane steering은 순간적으로
                    # +/-20 deg까지 튈 수 있으므로 raw steering으로 한 번에
                    # 전환하지 않는다. 마지막 유효 corner steering에서
                    # frame당 최대 몇 도씩만 white-lane steering 쪽으로 이동한다.
                    if (
                        steering is not None
                        and det["mode"] != "LOST"
                    ):
                        control_mode = "CORNER_LANE_FALLBACK"

                        fallback_delta = float(np.clip(
                            steering - last_corner_steering,
                            -CORNER_FALLBACK_STEER_STEP_DEG,
                            +CORNER_FALLBACK_STEER_STEP_DEG,
                        ))

                        control_steering = float(np.clip(
                            last_corner_steering + fallback_delta,
                            -MAX_STEERING_DEG,
                            +MAX_STEERING_DEG,
                        ))

                        # 다음 fallback frame의 기준점으로 갱신.
                        # yellow path가 다시 잡히면 위 corner_steering 분기로
                        # 즉시 복귀하고, white BOTH가 안정화되면 기존 exit
                        # 조건을 통해 NORMAL로 빠져나간다.
                        last_corner_steering = control_steering

                    else:
                        control_mode = "LOST"
                        control_steering = None

            # Preserve the original V8 lane/corner decision for the
            # next LiDAR update. Avoidance must not contaminate this hint.
            last_v8_control_mode = control_mode
            last_v8_control_steering = (
                float(control_steering)
                if control_steering is not None
                else None
            )

            # ------------------------------------------------
            # V13 dual-path corner avoidance arbitration
            # ------------------------------------------------
            # Straight sections keep V9 behavior.
            # In a sharp corner, avoidance follows the yellow reference path
            # and shifts it along the path-relative normal. The asphalt guard
            # may reduce that shift, but never invents a turn direction.
            if (
                AVOIDANCE_ENABLED
                and avoidance.state != "NORMAL"
            ):
                if corner_active:
                    if avoid_corner_steering is not None:
                        control_steering = avoid_corner_steering
                        control_mode = (
                            f"AVOID_{avoidance.state}_FRENET"
                        )
                    else:
                        # If the curved reference/road mask is unavailable,
                        # preserve the proven corner controller rather than
                        # falling back to a global x-offset that can leave road.
                        control_mode = (
                            f"AVOID_{avoidance.state}_CORNER_SAFE"
                        )

                elif (
                    avoidance.state in ("SHIFT_OUT", "PASS")
                    and avoid_lane_steering is not None
                ):
                    control_steering = avoid_lane_steering
                    control_mode = f"AVOID_{avoidance.state}"

                elif (
                    avoidance.state in ("SHIFT_IN", "RECOVER")
                    and avoid_lane_steering is not None
                    and control_mode != "LOST"
                ):
                    control_steering = avoid_lane_steering
                    control_mode = f"AVOID_{avoidance.state}"

            det["control_mode"] = control_mode
            det["avoid_state"] = avoidance.state
            det["avoid_direction"] = avoidance.avoid_direction
            det["avoid_near_ratio"] = avoid_near_ratio
            det["avoid_far_ratio"] = avoid_far_ratio

            if control_steering is not None:
                smoothed_steering = (
                    STEER_ALPHA * control_steering +
                    (1.0 - STEER_ALPHA) *
                    smoothed_steering
                )

                smoothed_steering = float(
                    np.clip(
                        smoothed_steering,
                        -MAX_STEERING_DEG,
                        MAX_STEERING_DEG,
                    )
                )

            else:
                smoothed_steering = 0.0

            # ------------------------------------------------
            # Preview-based path planning
            # ------------------------------------------------
            preview_heading, preview_curvature, preview_centers = (
                preview_lane_geometry(
                    det,
                    h,
                    w,
                    current_speed,
                )
            )

            det["preview_heading"] = preview_heading
            det["preview_curvature"] = preview_curvature
            det["preview_centers"] = preview_centers

            # Preview geometry is intentionally used ONLY for speed planning.
            # Steering output remains the same lane/corner steering that
            # produced the team's successful fast laps.
            steering_for_speed = control_steering

            if control_mode == "CORNER":
                # Do not force the car to crawl through every corner.
                # Let preview curvature decide the actual corner speed.
                target_speed = choose_target_speed(
                    det,
                    steering_for_speed,
                    heading_error,
                    preview_curvature,
                )

                target_speed = max(
                    target_speed,
                    SPEED_CORNER,
                )

            elif control_mode == "CORNER_HOLD":
                target_speed = SPEED_CORNER_HOLD

            elif control_mode == "CORNER_LANE_FALLBACK":
                target_speed = choose_target_speed(
                    det,
                    steering_for_speed,
                    heading_error,
                    preview_curvature,
                )

                # fallback은 드물게 발생하는 복구 상황이므로
                # 랩타임보다 안정성을 우선하되 완전히 정지하지는 않는다.
                target_speed = min(
                    max(target_speed, SPEED_MIN),
                    SPEED_CORNER_FALLBACK,
                )

            elif control_mode == "LOST":
                target_speed = 0.0

            else:
                target_speed = choose_target_speed(
                    det,
                    steering_for_speed if steering_for_speed is not None else None,
                    heading_error,
                    preview_curvature,
                )

            # ------------------------------------------------
            # V9.2 avoidance speed policy
            # ------------------------------------------------
            # Do NOT slow down merely because avoidance is active.
            #
            # - Any active avoidance state: keep 0.65 m/s cruise speed.
            # - This bypasses avoidance-induced LEFT_ONLY/RIGHT_ONLY and
            #   corner-state slowdowns.
            # - NORMAL straight sections are separately allowed up to 0.75 m/s.
            if (
                avoidance.state != "NORMAL"
                and control_steering is not None
            ):
                if corner_active:
                    # Curved avoidance needs time for the vehicle to follow the
                    # rotated path-normal offset. Do not force the old 0.65 m/s
                    # avoidance cruise through a sharp corner.
                    target_speed = min(
                        AVOID_CRUISE_SPEED,
                        AVOID_CORNER_GUARD_SPEED,
                    )
                else:
                    target_speed = AVOID_CRUISE_SPEED

            # ------------------------------------------------
            # V14 corner collision shield
            # ------------------------------------------------
            # If a cone is still directly ahead and the road guard cannot
            # produce a useful lateral shift, do not keep driving the original
            # corner reference path into the obstacle. Crawl, then stop at the
            # emergency threshold.
            shield_cap = corner_collision_shield_speed_cap(
                latest_cone_result,
                corner_active,
                avoid_corner_guard_state,
                avoid_corner_shift_scale,
            )
            det["corner_collision_shield_cap"] = shield_cap

            if shield_cap is not None:
                target_speed = min(target_speed, shield_cap)

            # A valid path that is explicitly moving away from the raw
            # obstacle is allowed to creep instead of entering a permanent
            # zero-speed deadlock.
            raw_escape_crawl = False
            raw_clearance = None
            if latest_raw_collision_cluster is not None:
                raw_clearance = _raw_cluster_lateral_clearance(
                    latest_raw_collision_cluster
                )
                raw_cy = float(
                    latest_raw_collision_cluster.get("cy", 0.0)
                )
                plan_side = det.get("corner_plan_side", "NONE")
                plan_away = (
                    (plan_side == "LEFT" and raw_cy < -AVOID_CENTER_DEADBAND_M)
                    or
                    (plan_side == "RIGHT" and raw_cy > AVOID_CENTER_DEADBAND_M)
                )
                raw_escape_crawl = bool(
                    corner_active
                    and avoidance.state in ("SHIFT_OUT", "PASS")
                    and plan_away
                    and avoid_corner_guard_state in ("FULL", "CLAMPED")
                    and float(avoid_corner_shift_scale) >= 0.35
                )

            raw_shield_cap = raw_collision_shield_speed_cap(
                latest_raw_collision_cluster,
                allow_escape_crawl=raw_escape_crawl,
            )
            det["raw_collision_shield_cap"] = raw_shield_cap
            det["raw_collision_clearance"] = raw_clearance
            det["raw_escape_crawl"] = raw_escape_crawl
            if raw_shield_cap is not None:
                target_speed = min(target_speed, raw_shield_cap)

            # Smooth the target itself so that a single noisy frame does not
            # immediately request a large speed change.
            if "smoothed_target_speed" not in locals():
                smoothed_target_speed = target_speed
            else:
                smoothed_target_speed = (
                    TARGET_SPEED_ALPHA * target_speed +
                    (1.0 - TARGET_SPEED_ALPHA) *
                    smoothed_target_speed
                )

            current_speed = ramp_speed(
                current_speed,
                smoothed_target_speed,
            )

            # V9.2: no avoidance-specific emergency crawl-speed override.
            # The original V8 drive/stop validity rules remain unchanged.

            if DRIVE_ENABLED:
                if control_steering is not None and target_speed > 0.0:
                    drive(
                        current_speed,
                        smoothed_steering,
                    )
                else:
                    current_speed = 0.0
                    drive(0.0, 0.0)

            panel = make_debug_panel(
                img,
                src_pts,
                bev,
                mask,
                det,
                (
                    smoothed_steering
                    if control_steering is not None
                    else None
                ),
                lateral_error,
                heading_error,
            )

            lane_width = det["lane_width_px"]
            lane_width_str = (
                f"{lane_width:.1f}px"
                if lane_width is not None
                else "None"
            )
            steer_str = (
                f"{smoothed_steering:+.1f}deg"
                if control_steering is not None
                else "None"
            )

            status = (
                f"RACING  "
                f"ctrl={control_mode}  "
                f"mode={det['mode']}  "
                f"conf={det['confidence']:.2f}  "
                f"left={det['left_count']}  "
                f"right={det['right_count']}  "
                f"width={lane_width_str}  "
                f"steer={steer_str}  "
                f"speed={current_speed:.2f}  "
                f"avoid={avoidance.state}/{avoidance.avoid_direction}  "
                f"reason={avoidance.direction_reason}  "
                f"guard={det.get('avoid_corner_guard_state', 'INACTIVE')} "
                f"gscale={det.get('avoid_corner_shift_scale', 0.0):.2f} "
                f"plan={det.get('corner_plan_side', 'NONE')} "
                f"L/R={det.get('corner_left_scale', 0.0):.2f}/"
                f"{det.get('corner_right_scale', 0.0):.2f}  "
                f"shift={det.get('corner_shift_ratio_cmd', 0.0):.2f} "
                f"rawcap={det.get('raw_collision_shield_cap')} "
                f"escape={int(bool(det.get('raw_escape_crawl', False)))} "
                f"offN={avoid_near_ratio:+.2f} "
                f"offF={avoid_far_ratio:+.2f}"
            )

            update_web(panel, status)

            now = time.time()

            if now - last_print >= PRINT_INTERVAL:
                print(
                    f"[LANE] "
                    f"ctrl={control_mode:<11} "
                    f"mode={det['mode']:<10} "
                    f"conf={det['confidence']:.2f} "
                    f"left={det['left_count']:<4} "
                    f"right={det['right_count']:<4} "
                    f"width={lane_width_str:<9} "
                    f"steer={steer_str:<10} "
                    f"speed={current_speed:.2f} "
                    f"target={target_speed:.2f} "
                    f"pcurve={det.get('preview_curvature', 0.0):.2f} "
                    f"ypath={len(corner_path)} "
                    f"avoid={avoidance.state:<9} "
                    f"dir={avoidance.avoid_direction:<5} "
                    f"reason={avoidance.direction_reason:<18} "
                    f"guard={det.get('avoid_corner_guard_state', 'INACTIVE'):<12} "
                    f"gscale={det.get('avoid_corner_shift_scale', 0.0):.2f} "
                    f"plan={det.get('corner_plan_side', 'NONE'):<5} "
                    f"L/R={det.get('corner_left_scale', 0.0):.2f}/"
                    f"{det.get('corner_right_scale', 0.0):.2f} "
                    f"shift={det.get('corner_shift_ratio_cmd', 0.0):.2f} "
                    f"rawcap={det.get('raw_collision_shield_cap')} "
                    f"rawclr={det.get('raw_collision_clearance')} "
                    f"escape={int(bool(det.get('raw_escape_crawl', False)))} "
                    f"offN={avoid_near_ratio:+.2f} "
                    f"offF={avoid_far_ratio:+.2f}"
                )

                last_print = now

            elapsed = time.time() - loop_start

            time.sleep(
                max(
                    0.0,
                    1.0 / TARGET_FPS - elapsed
                )
            )

    except KeyboardInterrupt:
        print("\n[MAIN] interrupted by Ctrl+C")

    finally:
        if DRIVE_ENABLED:
            stop_vehicle()

        stop_lidar_cluster_logger()
        stop_view()

        print("[MAIN] stopped safely")


if __name__ == "__main__":
    main()