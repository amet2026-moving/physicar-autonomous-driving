"""디버그 웹 뷰 (line-tracing.md 패턴과 동일 -- 튜닝/확인용). webui.html이 없으면 조용히
생략되고 주행에는 영향을 주지 않습니다.

DRIVE_ENABLED=True로 바꾸기 전에 반드시 이 뷰로 BEV/차선 인식이 맞는지 확인하세요
(config.py의 DRIVE_ENABLED 설명 참고).
"""
import os
import socket
import threading
import time

import cv2
import numpy as np

import versions

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEBUI_DIR = os.path.join(_THIS_DIR, "..", "assets", "line-tracing")

_frame = [b""]
_status = ["starting"]
_server = [None]
_start_time = [None]   # 기록측정 시작 시각 (mark_start() 호출 전엔 None)


def mark_start():
    """기록측정 시작. 신호등 대기가 끝나고 실제 주행 루프에 들어가기 직전에 호출하세요 --
    신호등 대기 시간은 랩타임에 안 들어가야 하므로 대기 전이 아니라 대기 후에 부릅니다."""
    _start_time[0] = time.time()
    print("기록측정 시작")


def elapsed_str():
    """기록측정 시작 이후 경과 시간. mark_start()를 아직 안 불렀으면 '--:--'."""
    if _start_time[0] is None:
        return "--:--"
    s = time.time() - _start_time[0]
    return f"{int(s) // 60:02d}:{s % 60:04.1f}"


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
        server = make_server("0.0.0.0", 5000, app, threaded=True)
    except (OSError, SystemExit):
        print("web view: port 5000 is in use -- 다른 예제 셀을 먼저 중지하세요")
        return None
    _server[0] = server
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("web view: open the MYAPP tab to watch")
    return server


def _draw_fit(img, fit, color, thickness=3):
    if fit is None:
        return
    h, w = img.shape[:2]
    pts = []
    for y in np.linspace(0, h - 1, 80):
        x = versions.lane.fit_x(fit, y)
        if x is not None and -50 <= x <= w + 50:
            pts.append((int(x), int(y)))
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_debug_panel(img, keeper):
    """카메라 / BEV / 흰색마스크 / 정보 4분할 디버그 패널. keeper는 versions.lane.LaneKeeper."""
    d = keeper.last_debug
    if not d:
        return img

    cam_vis = img.copy()
    cv2.polylines(cam_vis, [d["src_pts"].astype(np.int32)], True, (0, 255, 255), 2)

    det = d["det"]
    bev_vis = d["bev"].copy()
    _draw_fit(bev_vis, det["left_fit"], (0, 255, 0))
    _draw_fit(bev_vis, det["right_fit"], (0, 255, 0))
    contours, _ = cv2.findContours(d["yellow_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bev_vis, contours, -1, (255, 0, 255), 1)
    path = d["corner_path"]
    if path:
        pts = np.array([(int(x), int(y)) for x, y in path], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(bev_vis, [pts], False, (255, 0, 255), 3, cv2.LINE_AA)
    if det["center_near"] is not None:
        cv2.circle(bev_vis, (int(det["center_near"]), det["y_near"]), 8, (255, 255, 0), -1)
    if det["center_far"] is not None:
        cv2.circle(bev_vis, (int(det["center_far"]), det["y_far"]), 8, (255, 255, 0), -1)

    mask_vis = cv2.cvtColor(d["white_mask"], cv2.COLOR_GRAY2BGR)

    info = np.zeros_like(img)
    lane_width = det["lane_width_px"]
    lines = [
        f"mode: {det['mode']}  conf: {det['confidence']:.2f}",
        f"L={det['left_count']} R={det['right_count']}",
        f"lane_width_px: {lane_width:.1f}" if lane_width is not None else "lane_width_px: None",
        f"corner_active: {keeper.corner_active}",
        f"speed: {keeper.current_speed:.2f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(info, text, (12, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    top = np.hstack([cam_vis, bev_vis])
    bottom = np.hstack([mask_vis, info])
    return np.vstack([top, bottom])


def update_web(panel, status):
    ok, jpg = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        _frame[0] = jpg.tobytes()
    _status[0] = status


def serve():
    from flask import Flask, Response, jsonify
    app = Flask(__name__)
    webui_path = os.path.join(WEBUI_DIR, "webui.html")
    if not os.path.isfile(webui_path):
        print(f"web view: {webui_path} 없음 -- 디버그 패널 생략 (주행에는 영향 없음)")
        return None
    page = open(webui_path, encoding="utf-8").read()

    @app.get("/")
    def index():
        return page

    @app.get("/frame")
    def frame():
        return Response(_frame[0], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/data")
    def data():
        return jsonify({"status": _status[0]})

    return _start(app)
