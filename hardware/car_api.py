# PhysiCar 하드웨어/시뮬레이터 API를 감싼 얇은 HTTP 래퍼.
# camera(), lidar(), drive(), stop_vehicle(), set_camera_pose() 딱 이 다섯 개만 제공.
# 인식/필터링/상태 저장은 하지 않음 -- 차량과 통신하는 다른 모든 모듈은 반드시 이 파일을
# 통해서만 접근해야, 재시도/예외처리 방식이 호출부마다 제각각으로 흩어지지 않는다.
#
# 설계 원칙:
#   - drive()는 실패해도 예외를 삼키지 않고 그대로 던짐. 호출하는 메인 루프는
#     try/finally로 감싸서 finally에서 stop_vehicle()을 부르는 구조여야 함 -- 조용히
#     실패해서 마지막 명령을 계속 유지하는 것보다, 루프가 크게 멈추고 정지를 시도하는 게
#     더 안전함.
#   - stop_vehicle()은 반대로 무조건 예외를 삼킴 -- finally에서 '최후의 정지 시도'로
#     쓰이므로 여기서 또 예외가 나면 안 됨.
#   - set_camera_pose()는 성공/실패를 반환값(True/False)으로 알려줌 -- 호출부는 반드시
#     확인해야 함 (신호등 탐색 자세로 남은 채 주행이 시작되는 사고를 막기 위함).
import concurrent.futures
import math

import cv2
import numpy as np
import requests

from config import base as cfg

session = requests.Session()   # 매 프레임 새 TCP 연결을 맺지 않도록 세션 재사용
pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)   # 카메라/라이다를 동시에 요청하기 위한 스레드풀


def camera(width=None, height=None):
    """최신 카메라 프레임을 BGR ndarray로 반환. 실패하면 None."""
    try:
        params = {"width": width, "height": height} if width else {}   # 해상도 지정이 있으면 쿼리파라미터로 전달
        jpg = session.get(f"{cfg.BASE_URL}/camera", params=params, timeout=2).content   # timeout 단위: 초
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[car_api] camera() error: {e}")
        return None


def lidar(step=2):
    """[(각도(도, +=좌측), 거리(m)), ...] 리스트로 반환. 실패하면 빈 리스트.
    step: 각도 샘플링 간격 (도)."""
    try:
        data = session.get(f"{cfg.BASE_URL}/lidar", params={"step": step}, timeout=1).json()
        ranges = data.get("ranges", {})
        rmin = _to_float(data.get("range_min"), 0.02)   # 최소 유효거리 (m)
        rmax = _to_float(data.get("range_max"), 12.0)   # 최대 유효거리 (m)
        items = ranges.items() if isinstance(ranges, dict) else enumerate(ranges)

        points = []
        for key, d in items:
            d = _to_float(d)
            if d is None or not (rmin < d < rmax):   # 유효거리 범위 밖이면 버림
                continue
            if isinstance(ranges, dict):
                angle = _to_float(key)               # 키 자체가 각도(도)인 경우
            else:
                angle = key * step                   # 인덱스 * 간격으로 각도(도) 계산
                if angle > 180:
                    angle -= 360                      # -180~180 범위로 정규화
            if angle is None:
                continue
            points.append((angle, d))
        return points
    except Exception as e:
        print(f"[car_api] lidar() error: {e}")
        return []


def _to_float(x, default=None):
    """문자열/None 등을 안전하게 float으로 변환, 실패하면 default 반환."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def drive(speed, steering_deg):
    """PhysiCar 규약: speed 단위 m/s, steering_deg 단위 도(deg), +조향=좌회전.
    /steering, /speed는 별도의 HTTP POST라 원자적이지 않음 -- 하드웨어 API 자체의
    한계라 이 함수만으로는 고칠 수 없음. 실패 시 예외를 그대로 던진다(모듈 docstring 참고)."""
    r1 = session.post(f"{cfg.BASE_URL}/steering",
                       json={"value": math.radians(steering_deg)}, timeout=1)   # 라디안으로 변환해서 전송
    r1.raise_for_status()
    r2 = session.post(f"{cfg.BASE_URL}/speed",
                       json={"value": float(speed)}, timeout=1)   # 단위: m/s
    r2.raise_for_status()


def stop_vehicle():
    """최후의 정지 시도. 절대 예외를 밖으로 내보내지 않음 (main()의 finally에서 사용)."""
    try:
        drive(0.0, 0.0)
    except Exception:
        pass


def set_camera_pose(pan_deg, tilt_deg):
    """카메라 pan/tilt를 도(deg) 단위로 명령. 성공하면 True, 실패하면 False --
    반환값을 반드시 확인할 것 (카메라가 실제로는 탐색 자세에 남아있는 채로 주행이
    시작되는 사고를 막기 위함)."""
    try:
        pan_deg = float(np.clip(pan_deg, -30.0, 30.0))     # 하드웨어 pan 물리적 한계 (도)
        tilt_deg = float(np.clip(tilt_deg, -30.0, 30.0))   # 하드웨어 tilt 물리적 한계 (도)

        r1 = session.post(f"{cfg.BASE_URL}/camera/pan",
                           json={"value": math.radians(pan_deg)}, timeout=2)
        r1.raise_for_status()

        r2 = session.post(f"{cfg.BASE_URL}/camera/tilt",
                           json={"value": math.radians(tilt_deg)}, timeout=2)
        r2.raise_for_status()

        return True
    except Exception as e:
        print(f"[car_api] set_camera_pose() error: {e}")
        return False
