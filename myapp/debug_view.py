# 웹 디버그 뷰: PhysiCar webui.html 페이지를 서빙하고, 최신 프레임(주석달린 이미지)과
# 한줄 상태 텍스트를 그 페이지로 HTTP를 통해 전달한다.
#
# 아래 Flask 관련 부분(serve/update_web/stop_view)은 기존 검증된 코드를 그대로 옮긴
# 것이라 수정할 필요 없음. build_panel()만 비어있는데, sensors/camera.py,
# sensors/lidar.py, sensors/fusion.py가 각자의 관측 결과를 반환하게 되면, 그 결과들의
# 디버그 그림(원본/BEV/마스크/라이다 클러스터)을 한 이미지로 합치는 자리다.
import os
import socket
import threading

import cv2
import numpy as np

# PhysiCar 공식 webui.html 후보 경로들 -- 프로젝트 폴더 기준 상대경로(로컬 복사본을
# 쓰는 경우)를 먼저 보고, 없으면 실제 로봇에 설치된 PhysiCar SDK 절대경로를 본다.
# 실행 위치(cwd)에 의존하지 않도록 절대경로 후보를 추가함 -- 상대경로만 있으면
# physicar-autonomous-driving/ 밖(예: ~/physicar_ws/examples/...)에 있는 실제 파일을
# 못 찾는다.
WEBUI_DIR_CANDIDATES = [
    "assets/line-tracing",
    "examples/assets/line-tracing",
    os.path.expanduser("~/physicar_ws/examples/assets/line-tracing"),
]

_frame = [b""]           # 웹으로 내보낼 최신 프레임 (JPEG 인코딩된 바이트)
_status = ["starting"]    # 웹으로 내보낼 최신 상태 텍스트
_server = [None]          # 실행 중인 werkzeug 서버 인스턴스


def build_panel(camera_debug, lidar_debug, fusion_debug) -> np.ndarray:
    """TODO: sensors/camera.py, sensors/lidar.py, sensors/fusion.py가 만들어질 때까지
    미구현. 각 센서 모듈의 디버그 이미지를 하나의 패널로 합성해서 반환할 자리."""
    raise NotImplementedError


def update_web(panel: np.ndarray, status: str) -> None:
    """웹 뷰에 표시할 최신 프레임(panel)과 상태 문자열(status)을 갱신."""
    ok, jpg = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 82])   # JPEG 품질 0~100
    if ok:
        _frame[0] = jpg.tobytes()
    _status[0] = status


def stop_view() -> None:
    """실행 중인 웹 서버를 종료."""
    if _server[0]:
        try:
            _server[0].shutdown()
        except Exception:
            pass
        _server[0] = None


def _start(app):
    """Flask 앱을 백그라운드 스레드에서 5000번 포트로 구동."""
    import logging
    from werkzeug.serving import make_server

    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # 매 요청마다 찍히는 접속 로그 억제
    stop_view()   # 혹시 이전에 떠있던 서버가 있으면 먼저 정리

    try:
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("", 5000))   # 포트 5000이 비어있는지 미리 확인
        probe.close()
        server = make_server("0.0.0.0", 5000, app, threaded=True)
    except (OSError, SystemExit):
        print("[WEB] port 5000 is in use. Stop the other process and rerun.")
        return None

    _server[0] = server
    threading.Thread(target=server.serve_forever, daemon=True).start()   # 메인 루프를 막지 않도록 백그라운드 실행
    print("[WEB] debug view started on http://localhost:5000")
    return server


def serve():
    """웹 디버그 뷰 서버를 시작. main()에서 루프 시작 전 한 번만 호출."""
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    webui_path = None
    for candidate_dir in WEBUI_DIR_CANDIDATES:
        candidate_path = os.path.join(candidate_dir, "webui.html")
        if os.path.isfile(candidate_path):
            webui_path = candidate_path
            break

    if webui_path is None:
        tried = "\n".join(f"  - {os.path.join(d, 'webui.html')}" for d in WEBUI_DIR_CANDIDATES)
        raise FileNotFoundError(f"webui.html not found. Tried:\n{tried}")

    page = open(webui_path, "r", encoding="utf-8").read()   # 페이지 내용은 최초 1회만 읽어서 캐싱

    @app.get("/")
    def index():
        return page

    @app.get("/frame")
    def frame():
        # 첫 프레임이 도착하기 전에는 빈 이미지 대신 안내문구가 있는 이미지를 내려줌
        if not _frame[0]:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "waiting for first frame...", (12, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            ok, jpg = cv2.imencode(".jpg", blank)
            body = jpg.tobytes() if ok else b""
        else:
            body = _frame[0]
        return Response(body, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})   # 브라우저 캐시 방지

    @app.get("/data")
    def data():
        return jsonify({"status": _status[0]})

    return _start(app)
