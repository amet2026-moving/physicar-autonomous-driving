# 주행 중 rosbag 자동 녹화. 아직 미구현.
#
# 의도: 주행 루프가 시작될 때 녹화를 시작하고, utils/logger.py의 run_id()를 그대로
# 재사용해서 로그파일과 rosbag이 이름으로 짝지어지게 한다
# (logs/run_<id>.log <-> rosbags/run_<id>/).


def start_recording(run_id: str, topics: list[str]) -> None:
    """지정한 토픽들을 run_id 이름으로 녹화 시작. (topics: 녹화할 ROS 토픽 이름 목록)"""
    raise NotImplementedError


def stop_recording() -> None:
    """진행 중인 녹화를 종료."""
    raise NotImplementedError
