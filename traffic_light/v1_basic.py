"""신호등 인식 -- 팬/틸트로 랜덤 위치의 신호등을 탐색(SEARCH) -> 고정(LOCK) -> 초록 확인
대기(WAIT_GREEN) -> 주행 자세 복귀(RETURN_CAMERA). RED/UNKNOWN에서는 절대 출발하지 않고,
GREEN이 GREEN_CONFIRM_FRAMES 연속으로 확인된 뒤에만 카메라를 주행 자세로 되돌리고 주행을
허용합니다. main()은 이 모듈의 wait_for_green_and_return_to_driving_pose() 하나만 호출하면
됩니다 -- 주행 루프 시작 전에 한 번, 블로킹으로 실행됩니다.

초록/빨강 판정은 단순 HSV 색상만 보지 않고, "색상 후보가 검정 하우징(신호등 몸체)에
둘러싸여 있는지"를 같이 확인합니다 -- 배경 잔디의 초록색과 혼동되는 걸 막기 위함입니다.

[FIX] 원본(test_v8)은 RETURN_CAMERA 단계에서 set_camera_pose()의 반환값(성공/실패)을
무시하고 있었습니다. 명령이 네트워크 오류로 실패해도 "camera returned -> RACING enabled"를
찍고 그대로 주행을 시작해서, 카메라가 실제로는 서치 자세에 남아있는 채로 차선인식이 시작될
위험이 있었습니다 (코드 리뷰에서 확인). 여기서는 반환값을 확인해서 최대 3번 재시도하고,
그래도 실패하면 차량을 세워둔 채 SEARCH를 재개합니다 -- "카메라 자세를 확신 못 하는 채로
주행 시작" 자체를 원천 차단합니다.
"""
import time

import cv2
import numpy as np

import car_api
import config
import debug_view


# ============================================================
# 색상 판정
# ============================================================

def traffic_search_roi(frame):
    h, w = frame.shape[:2]
    x1n, y1n, x2n, y2n = config.TRAFFIC_ROI_NORM
    x1 = int(np.clip(x1n * w, 0, w - 1))
    y1 = int(np.clip(y1n * h, 0, h - 1))
    x2 = int(np.clip(x2n * w, x1 + 1, w))
    y2 = int(np.clip(y2n * h, y1 + 1, h))
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def traffic_make_masks(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(roi)

    red1 = cv2.inRange(
        hsv,
        np.array([config.TRAFFIC_RED_H1[0], config.TRAFFIC_RED_S_MIN, config.TRAFFIC_RED_V_MIN], dtype=np.uint8),
        np.array([config.TRAFFIC_RED_H1[1], 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(
        hsv,
        np.array([config.TRAFFIC_RED_H2[0], config.TRAFFIC_RED_S_MIN, config.TRAFFIC_RED_V_MIN], dtype=np.uint8),
        np.array([config.TRAFFIC_RED_H2[1], 255, 255], dtype=np.uint8))
    red_hsv = cv2.bitwise_or(red1, red2)

    ri, gi, bi = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
    red_dom = ((ri >= config.TRAFFIC_RED_CHANNEL_MIN) &
               ((ri - gi) >= config.TRAFFIC_RED_DOMINANCE) &
               ((ri - bi) >= config.TRAFFIC_RED_DOMINANCE)).astype(np.uint8) * 255
    red = cv2.bitwise_and(red_hsv, red_dom)

    green_hsv = cv2.inRange(
        hsv,
        np.array([config.TRAFFIC_GREEN_H[0], config.TRAFFIC_GREEN_S_MIN, config.TRAFFIC_GREEN_V_MIN], dtype=np.uint8),
        np.array([config.TRAFFIC_GREEN_H[1], 255, 255], dtype=np.uint8))
    green_dom = ((gi >= config.TRAFFIC_GREEN_CHANNEL_MIN) &
                 ((gi - ri) >= config.TRAFFIC_GREEN_DOMINANCE) &
                 ((gi - bi) >= config.TRAFFIC_GREEN_DOMINANCE)).astype(np.uint8) * 255
    green = cv2.bitwise_and(green_hsv, green_dom)

    k3 = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k3)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k3)
    return red, green, hsv


def traffic_dark_surround_ratio(hsv, contour):
    """색상 후보 둘레가 검정 하우징에 둘러싸여 있는 비율 -- 잔디 초록/배경 오탐 방지."""
    h, w = hsv.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    ew = max(bw + 4, int(round(bw * config.TRAFFIC_DARK_EXPAND)))
    eh = max(bh + 4, int(round(bh * config.TRAFFIC_DARK_EXPAND)))
    ex1, ey1 = max(0, int(round(cx - ew / 2))), max(0, int(round(cy - eh / 2)))
    ex2, ey2 = min(w, int(round(cx + ew / 2))), min(h, int(round(cy + eh / 2)))
    if ex2 <= ex1 or ey2 <= ey1:
        return 0.0

    ring = np.full((ey2 - ey1, ex2 - ex1), 255, dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= ex1
    shifted[:, 0, 1] -= ey1
    cv2.drawContours(ring, [shifted], -1, 0, thickness=-1)

    v = hsv[ey1:ey2, ex1:ex2, 2]
    valid = ring > 0
    n = int(np.count_nonzero(valid))
    if n < config.TRAFFIC_MIN_RING_PIXELS:
        return 0.0
    dark = (v <= config.TRAFFIC_DARK_V_MAX) & valid
    return float(np.count_nonzero(dark)) / float(n)


def traffic_candidate_score(area_ratio, fill, circularity, dark_ratio):
    area_score = float(np.clip(
        (area_ratio - config.TRAFFIC_MIN_BLOB_AREA_RATIO) /
        max(0.0025 - config.TRAFFIC_MIN_BLOB_AREA_RATIO, 1e-6), 0.0, 1.0))
    fill_score = float(np.clip(
        (fill - config.TRAFFIC_MIN_BBOX_FILL) /
        max(0.85 - config.TRAFFIC_MIN_BBOX_FILL, 1e-6), 0.0, 1.0))
    circularity_score = float(np.clip(
        (circularity - config.TRAFFIC_MIN_CIRCULARITY) /
        max(0.90 - config.TRAFFIC_MIN_CIRCULARITY, 1e-6), 0.0, 1.0))
    dark_score = float(np.clip(
        (dark_ratio - config.TRAFFIC_MIN_DARK_SURROUND_RATIO) /
        max(0.90 - config.TRAFFIC_MIN_DARK_SURROUND_RATIO, 1e-6), 0.0, 1.0))
    return 0.15 * area_score + 0.20 * fill_score + 0.15 * circularity_score + 0.50 * dark_score


def traffic_find_best_candidate(mask, hsv):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    roi_area = float(h * w)
    best = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        area_ratio = area / roi_area
        if not (config.TRAFFIC_MIN_BLOB_AREA_RATIO <= area_ratio <= config.TRAFFIC_MAX_BLOB_AREA_RATIO):
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if (x <= config.TRAFFIC_EDGE_MARGIN_PX or y <= config.TRAFFIC_EDGE_MARGIN_PX or
                x + bw >= w - config.TRAFFIC_EDGE_MARGIN_PX or y + bh >= h - config.TRAFFIC_EDGE_MARGIN_PX):
            continue

        bbox_area = float(max(1, bw * bh))
        fill = area / bbox_area
        if fill < config.TRAFFIC_MIN_BBOX_FILL:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < config.TRAFFIC_MIN_CIRCULARITY:
            continue

        dark_ratio = traffic_dark_surround_ratio(hsv, contour)
        if dark_ratio < config.TRAFFIC_MIN_DARK_SURROUND_RATIO:
            continue

        score = traffic_candidate_score(area_ratio, fill, circularity, dark_ratio)
        candidate = {"contour": contour, "bbox": (x, y, bw, bh), "area_ratio": area_ratio,
                     "fill": fill, "circularity": circularity, "dark_ratio": dark_ratio, "score": score}
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


class TrafficDetector:
    def __init__(self):
        self.green_streak = 0

    def reset(self):
        self.green_streak = 0

    def update(self, red_candidate, green_candidate):
        red_score = red_candidate["score"] if red_candidate is not None else 0.0
        green_score = green_candidate["score"] if green_candidate is not None else 0.0

        red_valid = red_score >= config.TRAFFIC_SCORE_MIN
        green_valid = (green_score >= config.TRAFFIC_SCORE_MIN and
                       green_score >= red_score + config.TRAFFIC_GREEN_OVER_RED_MARGIN)

        # 안전 우선순위: 신뢰할 만한 RED는 항상 출발을 막음
        if red_valid:
            raw_state = "RED"
            self.green_streak = 0
        elif green_valid:
            raw_state = "GREEN"
            self.green_streak += 1
        else:
            raw_state = "UNKNOWN"
            self.green_streak = 0

        confirmed_green = self.green_streak >= config.GREEN_CONFIRM_FRAMES
        return {"raw_state": raw_state, "confirmed_green": confirmed_green,
                "green_streak": self.green_streak, "red_score": red_score, "green_score": green_score}


# ============================================================
# 디버그 패널
# ============================================================

def traffic_draw_candidate(image, candidate, color, label):
    if candidate is None:
        return
    x, y, w, h = candidate["bbox"]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {candidate['score']:.2f}", (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def make_traffic_wait_panel(frame, roi, roi_box, red_mask, green_mask, red_candidate, green_candidate, result):
    h, w = frame.shape[:2]
    camera_vis = frame.copy()
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(camera_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(camera_vis, "WAIT_GREEN - traffic search ROI", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

    roi_vis = roi.copy()
    traffic_draw_candidate(roi_vis, red_candidate, (0, 0, 255), "RED")
    traffic_draw_candidate(roi_vis, green_candidate, (0, 255, 0), "GREEN")
    cv2.putText(roi_vis, "TRAFFIC ROI", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    red_vis = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    green_vis = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    half = max(1, w // 2)
    red_vis = cv2.resize(red_vis, (half, h), interpolation=cv2.INTER_NEAREST)
    green_vis = cv2.resize(green_vis, (w - half, h), interpolation=cv2.INTER_NEAREST)
    masks = np.hstack([red_vis, green_vis])
    cv2.putText(masks, "RED MASK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(masks, "GREEN MASK", (half + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    info = np.zeros_like(frame)
    raw = result["raw_state"]
    if raw == "RED":
        headline, color = "RED - WAIT", (0, 0, 255)
    elif raw == "GREEN":
        headline, color = "GREEN - CONFIRMING", (0, 255, 0)
    else:
        headline, color = "UNKNOWN - WAIT", (255, 255, 255)

    lines = [
        headline,
        f"green streak: {result['green_streak']}/{config.GREEN_CONFIRM_FRAMES}",
        f"red score: {result['red_score']:.3f}",
        f"green score: {result['green_score']:.3f}",
        "vehicle: STOPPED",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info, line, (18, 42 + i * 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 if i == 0 else 0.58, color if i == 0 else (235, 235, 235),
                    2 if i == 0 else 1, cv2.LINE_AA)

    roi_big = cv2.resize(roi_vis, (w, h), interpolation=cv2.INTER_AREA)
    top = np.hstack([camera_vis, roi_big])
    bottom = np.hstack([masks, info])
    return np.vstack([top, bottom])


def _traffic_detect_frame(img, detector):
    traffic_roi, traffic_roi_box = traffic_search_roi(img)
    red_mask, green_mask, hsv = traffic_make_masks(traffic_roi)
    red_candidate = traffic_find_best_candidate(red_mask, hsv)
    green_candidate = traffic_find_best_candidate(green_mask, hsv)
    result = detector.update(red_candidate, green_candidate)
    panel = make_traffic_wait_panel(img, traffic_roi, traffic_roi_box, red_mask, green_mask,
                                     red_candidate, green_candidate, result)
    return {
        "result": result, "panel": panel,
        "red_pixels": int(np.count_nonzero(red_mask)),
        "green_pixels": int(np.count_nonzero(green_mask)),
        "red_candidate": red_candidate, "green_candidate": green_candidate,
    }


def _traffic_print(prefix, pan_deg, tilt_deg, pack):
    result = pack["result"]
    print(f"[{prefix}] pan={pan_deg:+.0f} tilt={tilt_deg:+.0f} "
          f"state={result['raw_state']:<7} green={result['green_streak']}/{config.GREEN_CONFIRM_FRAMES} "
          f"red={result['red_score']:.3f} green_score={result['green_score']:.3f} "
          f"rpix={pack['red_pixels']:<5d} gpix={pack['green_pixels']:<5d}")


# ============================================================
# SEARCH -> LOCK -> WAIT_GREEN -> RETURN_CAMERA
# ============================================================

def _wait_green_at_locked_pose(detector, pan_deg, tilt_deg):
    """카메라를 고정한 채로 GREEN_CONFIRM_FRAMES 연속 GREEN을 기다림.
    True: 초록 확인됨. False: 신호등이 오래 안 보여서 SEARCH 재개 필요."""
    last_seen = time.time()
    last_print = 0.0
    print(f"[TRAFFIC] LOCKED at pan={pan_deg:+.1f} deg, tilt={tilt_deg:+.1f} deg -> WAIT_GREEN")

    while True:
        loop_start = time.time()
        img = car_api.camera()
        if img is None:
            if config.DRIVE_ENABLED:
                car_api.stop_vehicle()
            time.sleep(0.10)
            continue

        pack = _traffic_detect_frame(img, detector)
        result = pack["result"]
        if result["raw_state"] != "UNKNOWN":
            last_seen = time.time()

        status = (f"WAIT_GREEN LOCKED pan={pan_deg:+.0f} tilt={tilt_deg:+.0f} "
                  f"state={result['raw_state']} green={result['green_streak']}/{config.GREEN_CONFIRM_FRAMES} "
                  f"red={result['red_score']:.3f} green_score={result['green_score']:.3f}")
        debug_view.update_web(pack["panel"], status)

        now = time.time()
        if now - last_print >= config.PRINT_INTERVAL:
            _traffic_print("TRAFFIC LOCK", pan_deg, tilt_deg, pack)
            last_print = now

        if result["confirmed_green"]:
            print(f"[TRAFFIC] GREEN CONFIRMED ({config.GREEN_CONFIRM_FRAMES} frames) "
                  f"at pan={pan_deg:+.1f}, tilt={tilt_deg:+.1f}")
            return True

        # 1~2프레임 UNKNOWN만으로는 unlock하지 않음 -- 락 걸린 신호등이 실제로 오래 안
        # 보일 때만 SEARCH 재개
        if now - last_seen >= config.TRAFFIC_LOCK_LOST_SEC:
            print(f"[TRAFFIC] locked light lost for {config.TRAFFIC_LOCK_LOST_SEC:.1f}s -> resume SEARCH")
            detector.reset()
            return False

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, 1.0 / config.TARGET_FPS - elapsed))


def _return_camera_to_driving_pose():
    """[FIX] set_camera_pose()의 반환값을 확인해서 최대 3번 재시도. 그래도 실패하면 False를
    반환 -- 호출부는 절대 이 경우 주행을 시작하면 안 됨."""
    for attempt in range(3):
        if car_api.set_camera_pose(config.DRIVE_CAMERA_PAN_DEG, config.DRIVE_CAMERA_TILT_DEG):
            return True
        print(f"[TRAFFIC] RETURN_CAMERA attempt {attempt + 1}/3 failed, retrying...")
        time.sleep(0.3)
    return False


def wait_for_green_and_return_to_driving_pose():
    """대회 시작 전 블로킹 게이트. 차량은 정지 상태를 유지하며:
      SEARCH: 현재 정면부터 보고, 없으면 TRAFFIC_SEARCH_POSES를 순회.
      LOCK / WAIT_GREEN: RED/GREEN이 TRAFFIC_LOCK_FRAMES 연속 보이면 그 자세에 고정하고
        GREEN 확인을 기다림.
      RETURN_CAMERA: GREEN 확인 후 주행 자세(0,0)로 복귀 -- 복귀 실패 시 주행 시작하지 않고
        SEARCH를 재개함(위 FIX 참고).
    의도적으로 "너무 오래 걸리면 그냥 출발" 하는 타임아웃이 없습니다."""
    detector = TrafficDetector()
    pose_index = 0

    if config.DRIVE_ENABLED:
        car_api.stop_vehicle()

    print("[TRAFFIC] SEARCH_LIGHT start; vehicle remains STOPPED")

    while True:
        pan_deg, tilt_deg = config.TRAFFIC_SEARCH_POSES[pose_index]
        print(f"[TRAFFIC] move camera -> pose {pose_index + 1}/{len(config.TRAFFIC_SEARCH_POSES)} "
              f"pan={pan_deg:+.1f}, tilt={tilt_deg:+.1f}")

        car_api.set_camera_pose(pan_deg, tilt_deg)
        detector.reset()
        time.sleep(config.CAMERA_SETTLE_SEC)

        detect_started = time.time()
        lock_streak = 0
        lock_state = None
        last_print = 0.0
        locked = False

        while time.time() - detect_started < config.TRAFFIC_POSE_DWELL_SEC:
            loop_start = time.time()
            img = car_api.camera()
            if img is None:
                if config.DRIVE_ENABLED:
                    car_api.stop_vehicle()
                time.sleep(0.10)
                continue

            pack = _traffic_detect_frame(img, detector)
            result = pack["result"]
            raw_state = result["raw_state"]

            if raw_state != "UNKNOWN":
                if raw_state == lock_state:
                    lock_streak += 1
                else:
                    lock_state, lock_streak = raw_state, 1
            else:
                lock_state, lock_streak = None, 0

            status = (f"SEARCH_LIGHT pose={pose_index + 1}/{len(config.TRAFFIC_SEARCH_POSES)} "
                      f"pan={pan_deg:+.0f} tilt={tilt_deg:+.0f} state={raw_state} "
                      f"lock={lock_streak}/{config.TRAFFIC_LOCK_FRAMES} "
                      f"red={result['red_score']:.3f} green_score={result['green_score']:.3f}")
            debug_view.update_web(pack["panel"], status)

            now = time.time()
            if now - last_print >= config.PRINT_INTERVAL:
                _traffic_print("TRAFFIC SEARCH", pan_deg, tilt_deg, pack)
                last_print = now

            if lock_streak >= config.TRAFFIC_LOCK_FRAMES:
                locked = True
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, 1.0 / config.TARGET_FPS - elapsed))

        if locked:
            green = _wait_green_at_locked_pose(detector, pan_deg, tilt_deg)
            if green:
                print(f"[TRAFFIC] RETURN_CAMERA -> pan={config.DRIVE_CAMERA_PAN_DEG:+.1f}, "
                      f"tilt={config.DRIVE_CAMERA_TILT_DEG:+.1f}")

                camera_returned = _return_camera_to_driving_pose()

                if config.DRIVE_ENABLED:
                    car_api.stop_vehicle()
                time.sleep(config.CAMERA_SETTLE_SEC)

                if not camera_returned:
                    print("[TRAFFIC] RETURN_CAMERA FAILED after 3 attempts -> camera pose NOT "
                          "confirmed, vehicle remains STOPPED, resuming SEARCH")
                    detector.reset()
                    continue

                print("[TRAFFIC] camera returned -> RACING enabled")
                return

        # 이 자세에서 신호등을 못 봤거나, 락 걸렸던 신호등이 사라짐 -- 순환 탐색 계속.
        # 탐색이 오래 걸린다고 그냥 출발하지 않음.
        pose_index = (pose_index + 1) % len(config.TRAFFIC_SEARCH_POSES)
