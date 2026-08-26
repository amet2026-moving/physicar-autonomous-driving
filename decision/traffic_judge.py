# 신호등 시작 게이트: RED가 나타났다가 사라지는 것만으로 출발을 판단하는 RED-LOCK
# 방식. TTTTTT_physicar_ros2_red_lock_myapp.py의 TrafficLightStartDetector를 이
# 프로젝트의 인식/판단 분리에 맞게 이식한 것 -- 픽셀 단위 빨강 탐지(sensors/
# camera.py의 detect_red())는 인식이 전담하고, 여기는 "그 결과를 보고 언제 출발할지"
# 상태머신(_GateState)만 담당한다.
#
# GREEN은 보지 않는다: RED가 TRAFFIC_RED_CONFIRM_SEC 동안 지속되면 그 순간의
# 위치+크기를 락(lock)하고, 이후엔 그 락된 패치 안에서만(원래 면적/픽셀수 대비
# TRAFFIC_LOCK_RETAIN_AREA/PIXEL_RATIO 이하로) RED가 사라졌는지 확인한다 -- 다른
# 곳의 빨간 반사광/노이즈에 흔들리지 않기 위함. 락된 패치에서 RED가
# TRAFFIC_RED_RELEASE_SEC 동안 사라진 채로 유지되면 출발.
#
# 참고 파일에는 없지만, RED를 한 번도 못 찾고 영원히 대기하는 사고를 막기 위해
# NO_LOCK_TIMEOUT_SEC 폴백(예전 wait_for_green()에 있던 것과 같은 패턴)은 유지한다.
# 튜닝 상수는 config/camera_params.py에 있음.
import time

import cv2
import numpy as np
from enum import Enum

from config import base as cfg
from config import camera_params as cfg_cam
from hardware import car_api
from myapp import debug_view
from sensors import camera


class _GateState(Enum):
    WAIT_RED = "WAIT_RED"              # 아직 RED를 못 봤거나, 봤지만 확정 전
    RED_CONFIRMED = "RED_CONFIRMED"    # RED 위치+크기를 락하고 정지 유지 중
    RELEASE_VERIFY = "RELEASE_VERIFY"  # 락된 자리에서 RED가 사라짐 -- 지속시간 확인 중
    STARTED = "STARTED"                # 출발 확정


class _RedLockGate:
    """RED-LOCK 상태머신. update(frame)을 매 프레임 불러 상태를 진행시킨다.
    프레임 간 상태(state/lock_box/타이머)를 갖는다."""

    def __init__(self):
        self.state = _GateState.WAIT_RED
        self.red_begin_time = None       # WAIT_RED에서 RED가 시작된 시각 (time.monotonic)
        self.release_begin_time = None   # RED_CONFIRMED/RELEASE_VERIFY에서 RED가 사라진 시각
        self.lock_box = None             # 락된 bbox (ROI-local px), 락 전엔 None
        self.lock_reference_area = 0.0   # 락 당시 largest_area (락 유지 판정 기준값)
        self.lock_reference_pixels = 0   # 락 당시 red_pixels (락 유지 판정 기준값)

    def _lock_red_target(self, obs):
        """obs.bbox(ROI-local) 주변에 여유(padding)를 둔 락 박스를 만들고, 그 당시
        면적/픽셀수를 기준값으로 저장. bbox가 없으면 락 실패(False)."""
        if obs.bbox is None:
            return False
        x1, y1, x2, y2 = obs.bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad = max(
            cfg_cam.TRAFFIC_LOCK_PADDING_PX,
            int(round(max(bw, bh) * cfg_cam.TRAFFIC_LOCK_PADDING_RATIO)),
        )
        # camera.detect_red()의 focus_box 처리가 프레임 경계로 clamp해주므로 여기서는
        # 음수/범위초과를 신경쓰지 않아도 된다.
        self.lock_box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        self.lock_reference_area = max(float(obs.largest_area), cfg_cam.TRAFFIC_RED_MIN_AREA)
        self.lock_reference_pixels = max(int(obs.red_pixels), cfg_cam.TRAFFIC_LOCK_MIN_RED_PIXELS)
        print(
            f"[TL] RED TARGET LOCKED box={self.lock_box} "
            f"ref_area={self.lock_reference_area:.1f} ref_pixels={self.lock_reference_pixels}"
        )
        return True

    def _red_still_present(self, obs):
        """락 상태에서 '여전히 RED'로 볼지 판정. 원래 후보 대비 면적/픽셀수가
        RETAIN 비율 이상이어야 한다 -- 반사광 등 약한 잔상에 의한 오검출 억제."""
        area_threshold = max(
            cfg_cam.TRAFFIC_LOCK_MIN_RED_AREA,
            self.lock_reference_area * cfg_cam.TRAFFIC_LOCK_RETAIN_AREA_RATIO,
        )
        pixel_threshold = max(
            cfg_cam.TRAFFIC_LOCK_MIN_RED_PIXELS,
            int(round(self.lock_reference_pixels * cfg_cam.TRAFFIC_LOCK_RETAIN_PIXEL_RATIO)),
        )
        return (
            obs.valid_frame
            and obs.largest_area >= area_threshold
            and obs.red_pixels >= pixel_threshold
        )

    def update(self, frame) -> dict:
        """한 프레임 처리. 반환 dict: state, started(bool), red_detected, obs."""
        if self.state == _GateState.STARTED:
            return {"state": self.state, "started": True, "red_detected": False, "obs": None}

        now = time.monotonic()
        obs = camera.detect_red(frame, focus_box=self.lock_box)
        red_detected = self._red_still_present(obs) if self.lock_box is not None else obs.red_detected

        if self.state == _GateState.WAIT_RED:
            if red_detected:
                if self.red_begin_time is None:
                    self.red_begin_time = now
                if now - self.red_begin_time >= cfg_cam.TRAFFIC_RED_CONFIRM_SEC:
                    if self._lock_red_target(obs):
                        self.state = _GateState.RED_CONFIRMED
                        self.release_begin_time = None
                        print(f"[TL] WAIT_RED -> RED_CONFIRMED (red duration={now - self.red_begin_time:.3f}s)")
                    else:
                        self.red_begin_time = None
            else:
                self.red_begin_time = None

        elif self.state == _GateState.RED_CONFIRMED:
            if red_detected:
                self.release_begin_time = None
            elif obs.valid_frame:
                self.release_begin_time = now
                self.state = _GateState.RELEASE_VERIFY
                print("[TL] RED_CONFIRMED -> RELEASE_VERIFY (locked RED disappeared)")

        elif self.state == _GateState.RELEASE_VERIFY:
            if red_detected:
                self.release_begin_time = None
                self.state = _GateState.RED_CONFIRMED
                print("[TL] RELEASE_VERIFY -> RED_CONFIRMED (locked RED returned)")
            elif not obs.valid_frame:
                self.release_begin_time = now
            else:
                if self.release_begin_time is None:
                    self.release_begin_time = now
                release_duration = now - self.release_begin_time
                if release_duration >= cfg_cam.TRAFFIC_RED_RELEASE_SEC:
                    self.state = _GateState.STARTED
                    print(f"[TL] RELEASE_VERIFY -> STARTED (locked RED absent {release_duration:.3f}s)")

        return {
            "state": self.state,
            "started": self.state == _GateState.STARTED,
            "red_detected": red_detected,
            "obs": obs,
        }


# ============================================================
# 내부: 웹 디버그 패널 (myapp.debug_view.update_web()으로 내보냄)
# ============================================================

def _build_wait_panel(frame, roi_box, gate: _RedLockGate, result):
    camera_vis = frame.copy()
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(camera_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(camera_vis, "GATE - fixed ROI", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

    if gate.lock_box is not None:
        lx1, ly1, lx2, ly2 = gate.lock_box
        cv2.rectangle(camera_vis, (x1 + lx1, y1 + ly1), (x1 + lx2, y1 + ly2), (255, 0, 0), 2)
        cv2.putText(camera_vis, "RED LOCK", (x1 + lx1, max(18, y1 + ly1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    obs = result["obs"]
    if obs is not None and obs.bbox is not None:
        bx1, by1, bx2, by2 = obs.bbox
        color = (0, 0, 255) if result["red_detected"] else (0, 255, 255)
        cv2.rectangle(camera_vis, (x1 + bx1, y1 + by1), (x1 + bx2, y1 + by2), color, 2)

    state = result["state"]
    headline_color = {
        _GateState.WAIT_RED: (255, 255, 255),
        _GateState.RED_CONFIRMED: (0, 0, 255),
        _GateState.RELEASE_VERIFY: (0, 255, 0),
        _GateState.STARTED: (0, 255, 0),
    }[state]

    info = np.zeros_like(frame)
    lines = [
        state.value,
        f"lock: {'LOCKED' if gate.lock_box is not None else 'searching'}",
        f"red_detected: {result['red_detected']}",
        f"red_ratio: {obs.red_ratio:.4f}" if obs is not None else "red_ratio: -",
        f"largest_area: {obs.largest_area:.1f}" if obs is not None else "largest_area: -",
        f"valid_frame: {obs.valid_frame}" if obs is not None else "valid_frame: -",
        "vehicle: STOPPED",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info, line, (18, 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68 if i == 0 else 0.54, headline_color if i == 0 else (235, 235, 235),
                    2 if i == 0 else 1, cv2.LINE_AA)

    return np.hstack([camera_vis, info])


def _print_status(result):
    obs = result["obs"]
    print(
        f"[TL] {result['state'].value:<15} "
        f"lock={'YES' if result['obs'] is not None and result['obs'].bbox is not None else 'NO'} "
        f"red={result['red_detected']} "
        f"ratio={obs.red_ratio:.6f} area={obs.largest_area:.1f} valid={obs.valid_frame}"
        if obs is not None else f"[TL] {result['state'].value:<15} (no frame)"
    )


def _sleep_to_target_fps(loop_start):
    elapsed = time.time() - loop_start
    time.sleep(max(0.0, 1.0 / cfg.TARGET_FPS - elapsed))


# ============================================================
# 진입점
# ============================================================

def wait_for_departure() -> None:
    """주행 루프 시작 전, main.py에서 딱 한 번만 호출되는 블로킹 대기:
      1) 카메라를 고정 자세(DRIVE_CAMERA_PAN/TILT_DEG)로 맞춘다.
      2) 매 프레임 _RedLockGate.update()로 RED-LOCK 상태를 진행시킨다(차는 항상
         정지 상태).
      3) RED가 나타나 TRAFFIC_RED_CONFIRM_SEC 지속되면 락, 락된 자리에서 RED가
         TRAFFIC_RED_RELEASE_SEC 동안 사라져 있으면 출발.
      4) RED를 한 번도 못 찾은 채 NO_LOCK_TIMEOUT_SEC가 지나면 신호등을 아예 못
         찾은 것으로 보고 그냥 출발한다(stateless 폴백, 참고 파일엔 없는 안전장치).
    """
    car_api.stop_vehicle()

    if not car_api.set_camera_pose(cfg_cam.DRIVE_CAMERA_PAN_DEG, cfg_cam.DRIVE_CAMERA_TILT_DEG):
        print("[TL] camera pose command failed -- continuing with current pose")
    time.sleep(cfg_cam.CAMERA_SETTLE_SEC)

    print(f"[TL] RED-LOCK start gate active (ROI={cfg_cam.FIXED_ROI_NORM}); "
          "vehicle STOPPED; show RED to depart-check camera...")

    gate = _RedLockGate()
    search_started_at = time.time()
    last_print = 0.0

    while True:
        loop_start = time.time()

        frame = car_api.camera()
        if frame is None:
            car_api.stop_vehicle()
            time.sleep(0.10)
            continue

        result = gate.update(frame)

        depart = result["started"]
        depart_reason = "RED_RELEASED" if depart else None

        if not depart and gate.state == _GateState.WAIT_RED:
            if time.time() - search_started_at >= cfg_cam.NO_LOCK_TIMEOUT_SEC:
                print(f"[TL] no RED detected for {cfg_cam.NO_LOCK_TIMEOUT_SEC:.1f}s "
                      "-> departing anyway (stateless fallback)")
                depart = True
                depart_reason = "NO_RED_TIMEOUT"

        _, roi_box = camera.traffic_crop_roi(frame)
        panel = _build_wait_panel(frame, roi_box, gate, result)
        debug_view.update_web(panel, f"GATE state={result['state'].value} red={result['red_detected']}")

        now = time.time()
        if now - last_print >= cfg.PRINT_INTERVAL:
            _print_status(result)
            last_print = now

        if depart:
            print(f"[TL] DEPART -- reason={depart_reason}")
            return

        _sleep_to_target_fps(loop_start)
