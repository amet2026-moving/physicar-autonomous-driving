"""PhysiCar 하드웨어 API 래퍼 -- session, drive(), stop_vehicle(), camera(), lidar(),
set_camera_pose(). 이 파일만 팀 공통으로 통일해서 쓰면, 버전마다 제각각이던 예외처리/재시도
동작 차이(코드 리뷰에서 실제로 발견된 문제)가 한 곳으로 수렴합니다.

설계 결정 (리뷰에서 나온 이슈를 반영):
  - drive()는 실패 시 예외를 삼키지 않고 그대로 올립니다. auto4는 "서버 워치독(~1s)이 있어
    통신 실패해도 차가 알아서 멈춘다"는 가정 하에 조용히 무시했지만, 그 가정이 이 환경에서
    항상 보장된다는 근거가 없습니다. 대신 main.py의 while 루프를 try/finally로 감싸서, 예외가
    나면 finally에서 stop_vehicle()을 시도하고 루프를 크게(loudly) 종료하도록 합니다 --
    "조용히 실패해서 마지막 명령을 무한정 유지"하는 게 "루프가 멈추고 정지를 시도"하는 것보다
    위험하다는 게 이전 리뷰들에서 반복 확인된 패턴이라 그렇게 정했습니다.
  - camera()는 lidar()와 동일하게 넓은 Exception을 잡습니다 (원본은 requests.RequestException만
    잡아서 cv2.imdecode 실패 등이 새어나갈 수 있었습니다).
  - stop_vehicle()은 반대로 무조건 예외를 삼킵니다 -- main()의 finally에서 "최후의 정지 시도"로
    쓰이므로, 여기서 또 예외가 나면 안 됩니다.
"""
import concurrent.futures
import math

import cv2
import numpy as np
import requests

import config

session = requests.Session()   # 매 프레임 새 TCP 연결을 맺지 않도록 재사용
pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)   # 카메라/라이다 동시 요청용


def camera(width=None, height=None):
    """최신 카메라 프레임을 BGR ndarray로 반환. 실패 시 None."""
    try:
        params = {"width": width, "height": height} if width else {}
        jpg = session.get(f"{config.BASE_URL}/camera", params=params, timeout=2).content
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"camera() error: {e}")
        return None


def lidar(step=config.LIDAR_STEP_DEG):
    """[(각도(deg, +=좌측), 거리(m))] 리스트. 실패/이상한 응답이면 빈 리스트."""
    try:
        data = session.get(f"{config.BASE_URL}/lidar", params={"step": step}, timeout=1).json()
        ranges = data.get("ranges", {})
        rmin = _to_float(data.get("range_min"), 0.02)
        rmax = _to_float(data.get("range_max"), 12.0)
        items = ranges.items() if isinstance(ranges, dict) else enumerate(ranges)
        points = []
        for key, d in items:
            d = _to_float(d)
            if d is None or not (rmin < d < rmax):
                continue
            if isinstance(ranges, dict):
                angle = _to_float(key)
            else:
                angle = key * step
                if angle > 180:
                    angle -= 360
            if angle is None:
                continue
            points.append((angle, d))
        return points
    except Exception as e:
        print(f"lidar() error: {e}")
        return []


def _to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def drive(speed, steering_deg):
    """PhysiCar 규약: speed(m/s), steering_deg(도), +조향=좌.
    /speed, /steering은 별도 HTTP POST라 원자적이지 않습니다 -- 하드웨어 API 자체의 한계이며
    이 함수만으로는 고칠 수 없습니다. 실패 시 예외를 그대로 올립니다 (파일 상단 설명 참고)."""
    r1 = session.post(f"{config.BASE_URL}/steering",
                       json={"value": math.radians(steering_deg)}, timeout=1)
    r1.raise_for_status()
    r2 = session.post(f"{config.BASE_URL}/speed",
                       json={"value": float(speed)}, timeout=1)
    r2.raise_for_status()


def stop_vehicle():
    """최후의 정지 시도. 절대 예외를 밖으로 내보내지 않습니다 (main()의 finally에서 사용)."""
    try:
        drive(0.0, 0.0)
    except Exception:
        pass


def set_camera_pose(pan_deg, tilt_deg):
    """카메라 pan/tilt 명령(도). 성공하면 True, 실패하면 False -- 반환값을 반드시 확인해서
    쓰세요 (신호등 RETURN_CAMERA 단계에서 이 값을 무시하면 카메라가 실제로는 서치 자세에
    남아있는 채로 주행이 시작될 수 있습니다 -- 리뷰에서 발견된 버그, traffic_light.py 참고)."""
    try:
        pan_deg = float(np.clip(pan_deg, -30.0, 30.0))
        tilt_deg = float(np.clip(tilt_deg, -30.0, 30.0))

        r1 = session.post(f"{config.BASE_URL}/camera/pan",
                           json={"value": math.radians(pan_deg)}, timeout=2)
        r1.raise_for_status()

        r2 = session.post(f"{config.BASE_URL}/camera/tilt",
                           json={"value": math.radians(tilt_deg)}, timeout=2)
        r2.raise_for_status()

        return True
    except Exception as e:
        print(f"set_camera_pose() error: {e}")
        return False
