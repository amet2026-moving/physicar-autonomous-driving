# 실행 로그 저장 + 터미널 출력 담당.
# 모든 print() 출력을 터미널과 로그파일에 동시에 남기고(tee), 매 프레임 상태를
# `status += ...` 식으로 문자열을 덕지덕지 붙이는 대신 고정폭 한 줄로 깔끔하게 찍는다.
#
# 로그 파일 이름 규칙: logs/run_{YYYYMMDD}_{HHMMSS}.log
# 나중에 rosbag 자동저장(utils/rosbag_record.py)이 추가되면 같은 run_id를 재사용해서
# 로그파일과 rosbag이 파일명만으로 같은 실행인지 바로 매칭되게 할 것.
import atexit
import signal
import sys
from datetime import datetime
from pathlib import Path

_ORIGINAL_STDOUT = sys.stdout   # tee 걸기 전 원본 표준출력 (복원용으로 보관)
_ORIGINAL_STDERR = sys.stderr   # tee 걸기 전 원본 표준에러 (복원용으로 보관)
_RUN_LOG_HANDLE = None          # 현재 열려있는 로그 파일 핸들
_RUN_LOG_PATH = None            # 현재 로그 파일 경로
_LOG_CLOSED = False             # 로그를 이미 닫았는지 여부 (중복 종료 방지)
_RUN_ID = None                  # 이번 실행의 타임스탬프 ID (YYYYMMDD_HHMMSS)


class _TeeStream:
    """같은 텍스트를 터미널과 로그파일 양쪽에 동시에 쓰는 스트림."""

    def __init__(self, terminal, log_handle):
        self.terminal = terminal      # 원본 터미널 스트림
        self.log_handle = log_handle  # 로그 파일 핸들

    def write(self, message):
        self.terminal.write(message)
        self.log_handle.write(message)
        # 매 write마다 flush -- Ctrl+C나 강제종료 시 최근 로그가 유실되는 걸 최소화
        self.terminal.flush()
        self.log_handle.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_handle.flush()

    def isatty(self):
        try:
            return self.terminal.isatty()
        except Exception:
            return False

    def fileno(self):
        return self.terminal.fileno()


def run_id():
    """이번 실행의 타임스탬프 ID(YYYYMMDD_HHMMSS)를 반환. 로그파일과, 나중에 붙을
    rosbag이 이 값을 공유해서 같은 실행임을 파일명으로 알 수 있게 한다."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_ID


def setup_run_logging():
    """터미널+파일 동시 기록(tee)을 시작. 생성된 로그 파일 경로를 반환."""
    global _RUN_LOG_HANDLE, _RUN_LOG_PATH, _LOG_CLOSED

    if _RUN_LOG_HANDLE is not None:
        return _RUN_LOG_PATH   # 이미 시작했으면 중복 시작하지 않음

    filename = f"run_{run_id()}.log"

    candidates = [
        Path(__file__).resolve().parent.parent / "logs",   # 1순위: 프로젝트 루트의 logs/
        Path.cwd() / "logs",                                 # 2순위: 현재 작업 디렉토리의 logs/
    ]

    last_error = None
    for log_dir in candidates:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / filename
            handle = open(log_path, "a", encoding="utf-8", buffering=1)   # buffering=1: 줄단위 flush
            _RUN_LOG_HANDLE = handle
            _RUN_LOG_PATH = log_path
            _LOG_CLOSED = False
            sys.stdout = _TeeStream(_ORIGINAL_STDOUT, handle)   # 이 시점부터 print()가 파일에도 남음
            sys.stderr = _TeeStream(_ORIGINAL_STDERR, handle)
            atexit.register(close_run_logging)                  # 프로그램 종료 시 자동으로 닫히게 등록
            print(f"[LOG] saving to {log_path}")
            return log_path
        except Exception as e:
            last_error = e

    _ORIGINAL_STDERR.write(f"[LOG] WARNING: could not create run log: {last_error}\n")
    _ORIGINAL_STDERR.flush()
    return None


def close_run_logging():
    """현재 로그 파일을 flush하고 닫음. 여러 번 불러도 안전(한 번만 실행)."""
    global _RUN_LOG_HANDLE, _LOG_CLOSED

    if _LOG_CLOSED:
        return
    _LOG_CLOSED = True

    try:
        if _RUN_LOG_HANDLE is not None:
            print(f"[LOG] closed -> {_RUN_LOG_PATH}")
            sys.stdout.flush()
            sys.stderr.flush()
    except Exception:
        pass

    try:
        sys.stdout = _ORIGINAL_STDOUT   # 표준출력/에러를 원래대로 복원
        sys.stderr = _ORIGINAL_STDERR
    except Exception:
        pass

    try:
        if _RUN_LOG_HANDLE is not None:
            _RUN_LOG_HANDLE.flush()
            _RUN_LOG_HANDLE.close()
    except Exception:
        pass

    _RUN_LOG_HANDLE = None


def _handle_termination_signal(signum, _frame):
    """SIGTERM을 KeyboardInterrupt로 바꿔서, 기존 try/finally 정리 코드가 그대로 실행되게 한다."""
    try:
        print(f"\n[MAIN] termination signal received: {signum}")
    finally:
        raise KeyboardInterrupt


def install_shutdown_handlers():
    """SIGTERM을 잡아서 Ctrl+C와 동일하게 처리되게 설치. SIGKILL/전원차단은
    파이썬 코드로 잡을 수 없으므로 대상이 아님."""
    try:
        signal.signal(signal.SIGTERM, _handle_termination_signal)
    except Exception as e:
        print(f"[LOG] SIGTERM handler unavailable: {e}")


def status_line(elapsed_str, mode, traffic, lane, obstacle, steer_deg, speed, hz, extra=""):
    """매 프레임(또는 heartbeat)마다 고정폭 한 줄로 상태를 출력.

    형식 예: [00:12.3] MODE=LANE_FOLLOW      light=GREEN  lane=CORNER   obstacle=CLEAR  steer=+12.3  speed=0.55   9.8Hz

    elapsed_str : 경과시간 문자열 (분:초.소수 형식)
    mode        : VehicleMode
    traffic     : TrafficLightState
    lane        : LaneState
    obstacle    : ObstacleState
    steer_deg   : 조향각 (도, deg)
    speed       : 속도 (m/s)
    hz          : 이번 프레임 처리 주파수 (Hz)
    extra       : 추가로 붙일 문자열 (선택)
    """
    line = (
        f"[{elapsed_str}] "
        f"MODE={mode.value:<15} "
        f"light={traffic.value:<8} "
        f"lane={lane.value:<10} "
        f"obstacle={obstacle.value:<6} "
        f"steer={steer_deg:+6.1f} "     # 도(deg), 부호 표시
        f"speed={speed:5.2f} "          # m/s
        f"{hz:5.1f}Hz"                   # 초당 프레임 처리 횟수
    )
    if extra:
        line += f"  {extra}"
    print(line)
