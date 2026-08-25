# 녹화된 rosbag을 재생해서 오프라인으로 디버깅하기 위한 모듈. 아직 미구현.
#
# 의도: 실제 주행 때 쓰는 sensors/ 인터페이스와 동일한 형태로 재생해서, 실차 없이도
# decision/, control/ 로직을 기록된 주행 데이터로 테스트할 수 있게 한다.


def replay(bag_path: str) -> None:
    """지정한 rosbag 파일을 재생. (bag_path: rosbag 파일/디렉토리 경로)"""
    raise NotImplementedError
