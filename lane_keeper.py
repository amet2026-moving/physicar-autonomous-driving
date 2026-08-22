"""LaneKeeper -- 차선 인식(lane_tracing.py) 결과를 받아 코너모드 진입/이탈, 조향 EMA, 속도계획을
조합하는 상태 있는 컨트롤러. main()에서 한 번만 만들어서 루프 내내 재사용하고, 매 프레임
keeper.step(img) -> (line_steer, speed, curve_direction, status) 를 호출합니다.
장애물 회피(obstacle_avoidance.py)는 여기서 하지 않고 main.py에서 line_steer에 더합니다.

[FIX] 원본(auto4)의 코너진입 디바운스(CORNER_ENTER_WEAK_WHITE_FRAMES)는 weak_white(흰선
LEFT_ONLY/RIGHT_ONLY/LOST) 조건에만 걸려있고, normal_sharp/normal_big_steer(조향각/헤딩
기반)는 단일 프레임만으로도 즉시 코너모드를 켤 수 있었습니다. 콘이 흰선을 살짝 왜곡시켜
폴리핏이 한 프레임 튀면 이 경로로 여전히 오진입할 수 있어서(코드 리뷰에서 확인), 이 두
조건에도 CORNER_ENTER_DEBOUNCE_FRAMES 프레임 연속 디바운스를 추가했습니다.
"""
import cv2
import numpy as np

import config
import lane_tracing


class LaneKeeper:
    """흰색 양쪽 경계선(평상시, Stanley 조향) / 노란점선경로(급커브) 하이브리드 차선 추종.
    상태를 인스턴스에 들고 있으므로 main()에서 한 번만 만들어서 루프 내내 재사용합니다."""

    def __init__(self):
        self.tracker = lane_tracing.LaneTracker()
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
                steer = lane_tracing.lane_model_steer(img)
                self.current_speed = lane_tracing.ramp_speed(self.current_speed, config.SPEED_MAX * 0.7)
                if abs(steer) > 1.0:
                    self.last_steer_sign = 1.0 if steer > 0 else -1.0
                curve_direction = (1.0 if steer > 0 else -1.0) if abs(steer) >= config.LINE_CURVE_THRESHOLD else None
                return steer, self.current_speed, curve_direction, f"learned steer {steer:+.1f}"
            except Exception as e:
                print(f"lane model error: {e} -- BEV 파이프라인으로 폴백")

        h, w = img.shape[:2]
        src_pts, M = lane_tracing.build_bev_matrices(w, h)
        bev = cv2.warpPerspective(img, M, (w, h))

        white_mask = lane_tracing.make_white_mask(bev)
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

        steering, lateral_error, heading_error = lane_tracing.steering_from_lane(det_for_steer, w, self.current_speed)
        if lateral_error is not None:
            self.smoothed_lateral_error = (config.LANE_BIAS_ALPHA * lateral_error +
                                            (1.0 - config.LANE_BIAS_ALPHA) * self.smoothed_lateral_error)

        preview_heading, preview_curvature, _ = lane_tracing.preview_lane_geometry(det, h, w, self.current_speed)

        yellow_mask = lane_tracing.make_yellow_corner_mask(bev)
        yellow_pts = lane_tracing.yellow_centroids(yellow_mask)
        corner_path = lane_tracing.build_corner_path(yellow_pts, w, h)
        corner_target = lane_tracing.choose_corner_target(corner_path, w, h)
        corner_steering = lane_tracing.steering_from_corner_target(corner_target, w, h)

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
            target_speed = lane_tracing.choose_target_speed(det, steering_for_speed, heading_error, preview_curvature)

        self.smoothed_target_speed = target_speed if self.smoothed_target_speed is None else (
            config.TARGET_SPEED_ALPHA * target_speed + (1.0 - config.TARGET_SPEED_ALPHA) * self.smoothed_target_speed)
        self.current_speed = lane_tracing.ramp_speed(self.current_speed, self.smoothed_target_speed)

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
