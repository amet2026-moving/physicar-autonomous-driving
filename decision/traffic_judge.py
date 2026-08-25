# 신호등 원시 인식 결과(sensors.camera.TrafficLightObservation)를 TrafficLightState로
# 판정하고, 주행 루프 시작 전 신호대기 게이트를 담당하는 모듈.
#
# 인식(ROI 크롭 + 색상 마스크 + 위치 락)은 sensors/camera.py의 detect_traffic_light()가
# 전담한다 -- 이 파일은 "그 점수를 보고 RED/GREEN/UNKNOWN 중 뭐라고 볼지"와 "언제
# 출발할지"만 판단한다. wait_for_green()의 디바운스(GREEN_CONFIRM_FRAMES/
# RED_CLEAR_DEPART_SEC)와 NO_LOCK_TIMEOUT_SEC 폴백은 light_1.py의
# StartGateController/wait_for_start_gate()를 그대로 옮긴 것. 튜닝 상수는
# config/camera_params.py에 있음.
import time

import cv2
import numpy as np

from config import base as cfg
from config import camera_params as cfg_cam
from hardware import car_api
from myapp import debug_view
from sensors import camera
from utils.states import TrafficLightState


def is_green(traffic_obs) -> bool:
    """GREEN 확정 조건: 최소 점수(TRAFFIC_SCORE_MIN) 이상이면서 RED보다
    TRAFFIC_GREEN_OVER_RED_MARGIN 이상 앞서야 함 -- 두 색이 동시에 애매하게 걸리는
    프레임(노이즈로 겹쳐 보일 때)에 성급히 GREEN으로 확정하지 않기 위한 여유."""
    return (
        traffic_obs.green_score >= cfg_cam.TRAFFIC_SCORE_MIN
        and traffic_obs.green_score >= traffic_obs.red_score + cfg_cam.TRAFFIC_GREEN_OVER_RED_MARGIN
    )


def judge_traffic_light(traffic_obs) -> TrafficLightState:
    """RED가 최소 점수 이상이면 GREEN과 동시에 걸려도 RED 우선(정지 유지).
    그 외엔 is_green()으로 GREEN/UNKNOWN을 가른다."""
    if traffic_obs.red_score >= cfg_cam.TRAFFIC_SCORE_MIN:
        return TrafficLightState.RED
    if is_green(traffic_obs):
        return TrafficLightState.GREEN
    return TrafficLightState.UNKNOWN


# ============================================================
# 내부: 위치 락 이후의 RED/GREEN 디바운스 (프레임 간 상태를 들고 있음)
# ============================================================

class _StartGateController:
    """RED->정지 유지. RED가 사라지면 RED_CLEAR_DEPART_SEC 후 출발, 또는 그 전에
    GREEN이 GREEN_CONFIRM_FRAMES 연속 확인되면 즉시 출발 -- 둘 중 먼저 오는 쪽."""

    def __init__(self):
        self.green_streak = 0
        self.was_red = False
        self.red_cleared_at = None

    def reset(self):
        self.__init__()

    def update(self, traffic_obs) -> dict:
        state = judge_traffic_light(traffic_obs)

        if state == TrafficLightState.RED:
            self.was_red = True
            self.red_cleared_at = None
            self.green_streak = 0
        else:
            # RED가 최소 한 번은 확인된 뒤에 사라진 경우에만 "사라짐" 타이머를 켠다 --
            # 카메라 초기화 지연 등으로 첫 프레임부터 UNKNOWN이 나오면 곧바로
            # 출발해버리는 걸 방지.
            if self.was_red and self.red_cleared_at is None:
                self.red_cleared_at = time.time()
            self.green_streak = self.green_streak + 1 if state == TrafficLightState.GREEN else 0

        red_cleared_sec = (
            time.time() - self.red_cleared_at if self.red_cleared_at is not None else None
        )
        depart_by_green = self.green_streak >= cfg_cam.GREEN_CONFIRM_FRAMES
        depart_by_red_clear = (
            red_cleared_sec is not None and red_cleared_sec >= cfg_cam.RED_CLEAR_DEPART_SEC
        )

        depart_reason = None
        if depart_by_green:
            depart_reason = "GREEN_CONFIRMED"
        elif depart_by_red_clear:
            depart_reason = "RED_CLEARED"

        return {
            "state": state,
            "depart": depart_by_green or depart_by_red_clear,
            "depart_reason": depart_reason,
            "green_streak": self.green_streak,
            "red_cleared_sec": red_cleared_sec,
        }


# ============================================================
# 내부: 웹 디버그 패널 (myapp.debug_view.update_web()으로 내보냄)
# ============================================================
# 주행 루프 시작 전 게이트 단계라 myapp.debug_view.build_panel()(메인 루프의
# camera/lidar/fusion 통합 패널)과는 별개로, 이 단계 전용의 가벼운 패널을 직접
# 만들어 내보낸다 -- sensors/camera.py의 __main__ 자체 테스트가 하는 것과 같은 방식.

def _bbox_to_frame(bbox, roi_box):
    """sensors.camera가 돌려주는 bbox는 ROI 로컬 좌표라, 원본 프레임에 그리려면
    ROI 원점(roi_box의 x1,y1)만큼 옮겨야 한다."""
    if bbox is None:
        return None
    x, y, w, h = bbox
    x1, y1, _, _ = roi_box
    return (x + x1, y + y1, w, h)


def _draw_bbox(img, bbox, color, label, score):
    if bbox is None:
        return
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"{label} {score:.2f}", (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _build_wait_panel(frame, roi_box, traffic_obs, status):
    camera_vis = frame.copy()
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(camera_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(camera_vis, "GATE - fixed ROI", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

    if traffic_obs.locked_bbox is not None:
        lx, ly, lw, lh = _bbox_to_frame(traffic_obs.locked_bbox, roi_box)
        cv2.rectangle(camera_vis, (lx, ly), (lx + lw, ly + lh), (0, 255, 255), 2)
        cv2.putText(camera_vis, "LOCKED", (lx, max(18, ly - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    _draw_bbox(camera_vis, _bbox_to_frame(traffic_obs.red_bbox, roi_box), (0, 0, 255), "RED", traffic_obs.red_score)
    _draw_bbox(camera_vis, _bbox_to_frame(traffic_obs.green_bbox, roi_box), (0, 255, 0), "GREEN", traffic_obs.green_score)

    info = np.zeros_like(frame)
    label = status["raw_label"]
    if label == "LOCKING":
        headline, color = "LOCKING POSITION...", (0, 255, 255)
    elif label == "RED":
        headline, color = "RED - WAIT", (0, 0, 255)
    elif label == "GREEN":
        headline, color = "GREEN - CONFIRMING", (0, 255, 0)
    else:
        headline, color = "UNKNOWN - WAIT", (255, 255, 255)

    lock_line = (
        "lock: LOCKED" if traffic_obs.locked
        else f"lock: searching ({traffic_obs.lock_streak}/{cfg_cam.POSITION_LOCK_FRAMES})"
    )
    red_cleared_str = (
        f"{status['red_cleared_sec']:.2f}s" if status.get("red_cleared_sec") is not None else "-"
    )

    lines = [
        headline,
        lock_line,
        f"green streak: {status['green_streak']}/{cfg_cam.GREEN_CONFIRM_FRAMES}",
        f"red score: {traffic_obs.red_score:.3f}",
        f"green score: {traffic_obs.green_score:.3f}",
        f"red cleared for: {red_cleared_str}",
        f"depart reason: {status.get('depart_reason') or '-'}",
        "vehicle: STOPPED",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info, line, (18, 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68 if i == 0 else 0.54, color if i == 0 else (235, 235, 235),
                    2 if i == 0 else 1, cv2.LINE_AA)

    return np.hstack([camera_vis, info])


def _print_status(status, traffic_obs):
    red_cleared_str = (
        f"{status['red_cleared_sec']:.2f}" if status.get("red_cleared_sec") is not None else "-"
    )
    print(
        f"[TRAFFIC] state={status['raw_label']:<7} "
        f"lock={'YES' if traffic_obs.locked else 'NO'}({traffic_obs.lock_streak}/{cfg_cam.POSITION_LOCK_FRAMES}) "
        f"green={status['green_streak']}/{cfg_cam.GREEN_CONFIRM_FRAMES} "
        f"red_cleared={red_cleared_str:<5} "
        f"red_score={traffic_obs.red_score:.3f} "
        f"green_score={traffic_obs.green_score:.3f}"
    )


def _sleep_to_target_fps(loop_start):
    elapsed = time.time() - loop_start
    time.sleep(max(0.0, 1.0 / cfg.TARGET_FPS - elapsed))


# ============================================================
# 진입점
# ============================================================

def wait_for_green() -> None:
    """주행 루프 시작 전, main.py에서 딱 한 번만 호출되는 블로킹 대기 (light_1.py의
    wait_for_start_gate()와 동일한 흐름):
      1) 카메라를 고정 자세(DRIVE_CAMERA_PAN/TILT_DEG)로 맞춘다.
      2) 매 프레임 sensors.camera.detect_traffic_light()로 위치 락 + 점수를 받는다 --
         락 전에는 STOP/출발 판정을 하지 않음(차는 항상 정지 상태).
      3) 락 상태에서 RED면 정지 유지. RED가 사라지면 RED_CLEAR_DEPART_SEC 후 출발,
         또는 그 전에 GREEN이 GREEN_CONFIRM_FRAMES 연속 확인되면 즉시 출발.
      4) 락이 풀리면(POSITION_LOCK_LOST_SEC 동안 미검출) 디바운스 상태를 리셋하고
         다시 락 탐색 단계로 돌아간다(sensors.camera가 내부적으로 재탐색함).
      5) 시작 후 한 번도 락에 성공 못한 채 NO_LOCK_TIMEOUT_SEC가 지나면 신호등을
         아예 못 찾은 것으로 보고 그냥 출발한다(stateless 폴백).
    """
    car_api.stop_vehicle()

    if not car_api.set_camera_pose(cfg_cam.DRIVE_CAMERA_PAN_DEG, cfg_cam.DRIVE_CAMERA_TILT_DEG):
        print("[TRAFFIC] camera pose command failed -- continuing with current pose")
    time.sleep(cfg_cam.CAMERA_SETTLE_SEC)

    print(f"[TRAFFIC] fixed-ROI start gate active (ROI={cfg_cam.FIXED_ROI_NORM}); "
          "vehicle STOPPED; locking position...")

    controller = _StartGateController()
    ever_locked = False
    was_locked = False
    search_started_at = time.time()
    last_print = 0.0

    while True:
        loop_start = time.time()

        frame = car_api.camera()
        if frame is None:
            car_api.stop_vehicle()
            time.sleep(0.10)
            continue

        traffic_obs = camera.detect_traffic_light(frame)

        if traffic_obs.locked:
            ever_locked = True
            was_locked = True
            status = controller.update(traffic_obs)
            status["raw_label"] = status["state"].value
        else:
            if was_locked:
                print("[TRAFFIC] lock lost -> re-scanning full ROI")
                controller.reset()
            was_locked = False

            status = {
                "raw_label": "LOCKING",
                "depart": False,
                "depart_reason": None,
                "green_streak": 0,
                "red_cleared_sec": None,
            }

            # 신호등을 아예 못 찾는 상태로 계속 있으면 영원히 대기하게 되므로,
            # 일정 시간 뒤엔 강제로 출발한다(stateless 주행 요건상 이게 더 안전).
            if not ever_locked and time.time() - search_started_at >= cfg_cam.NO_LOCK_TIMEOUT_SEC:
                print(f"[TRAFFIC] no object locked for {cfg_cam.NO_LOCK_TIMEOUT_SEC:.1f}s "
                      "-> departing anyway (stateless fallback)")
                status["depart"] = True
                status["depart_reason"] = "NO_LOCK_TIMEOUT"

        _, roi_box = camera.traffic_crop_roi(frame)
        panel = _build_wait_panel(frame, roi_box, traffic_obs, status)
        debug_view.update_web(
            panel,
            f"GATE state={status['raw_label']} lock={'YES' if traffic_obs.locked else 'NO'}",
        )

        now = time.time()
        if now - last_print >= cfg.PRINT_INTERVAL:
            _print_status(status, traffic_obs)
            last_print = now

        if status["depart"]:
            print(f"[TRAFFIC] DEPART -- reason={status['depart_reason']}")
            return

        _sleep_to_target_fps(loop_start)
