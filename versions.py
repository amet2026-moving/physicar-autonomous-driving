"""사용할 버전을 여기서 선택합니다.

아래 블록 중 주석 처리 안 된(활성화된) 블록이 지금 실제로 쓰이는 조합입니다.
새로 테스트할 때는 새 블록을 맨 아래에 추가하고 그 블록만 주석 해제하세요 -- 이전 블록들은
주석 처리된 채로 남겨두면 그게 곧 주행 기록이 됩니다 (날짜/랩타임/특징을 헤더에 적어두세요).
주석 활성화 : ctrl + /

규칙:
  - 반드시 블록 하나만 활성화(주석 해제) 상태여야 합니다. 두 블록이 동시에 활성화되면
    아래쪽 블록의 CONFIG_VERSION/LANE_VERSION/OBSTACLE_VERSION/TRAFFIC_VERSION 값이
    그냥 덮어써서 그게 최종 적용됩니다 -- 둘 다 남겨두면 헷갈리니 위쪽은 꼭 주석 처리하세요.
  - 각 블록 헤더의 필드: 설명(뭘 바꿨는지), 점수(랩타임/초), 충돌(횟수), 이탈(트랙 이탈 횟수).
  - CONFIG/LANE/OBSTACLE/TRAFFIC 네 값은 서로 독립적으로 조합 가능합니다. 예를 들어
    LANE_VERSION만 새 버전으로 바꾸고 나머지는 그대로 둬도 됩니다 -- 파일 이름(확장자 제외)만
    정확히 일치하면 됩니다 (예: lane/v2_bev.py를 쓰려면 LANE_VERSION = "v2_bev").
"""
import importlib
import sys

"""0823 다현
설명: 재모
modi_test_v5/v6, Sangheon test_v7/v8, 성찬 auto3/auto4 버전 통합 및 자잘한 수정
점수: 63초
충돌: 0회
이탈: 1회
"""
#CONFIG_VERSION = "v1_basic"
#LANE_VERSION = "v1_basic"
#OBSTACLE_VERSION = "v1_basic"
#TRAFFIC_VERSION = "v1_basic"
 
"""0824 윤지
설명: 새 알고리즘 test
점수: 초
충돌: 회
이탈: 회
"""
CONFIG_VERSION = "v1_basic"
LANE_VERSION = "v1_basic"
OBSTACLE_VERSION = "v1_basic"
TRAFFIC_VERSION = "v1_basic"

"""0824 수민
설명: 3_1 장애물 회피 개선
점수: 초
충돌: 회
이탈: 회
"""
#CONFIG_VERSION = "v1_basic"
#LANE_VERSION = "v1_basic"
#OBSTACLE_VERSION = "v3_sumin_test"
#TRAFFIC_VERSION = "v1_basic"




# 이 아래로는 작성하지 말하주세요.
# (여기서부터는 위에서 고른 버전 이름을 실제로 불러오는 코드입니다)
config = importlib.import_module(f"config.{CONFIG_VERSION}")
sys.modules["config"] = config

# 나머지 셋은 별칭 등록이 필요 없습니다 -- main.py가 lane_tracing.LaneKeeper() 대신
# versions.lane.LaneKeeper() 식으로 이 변수를 직접 통해서 접근하기 때문입니다.
lane = importlib.import_module(f"lane.{LANE_VERSION}")
obstacle = importlib.import_module(f"obstacle.{OBSTACLE_VERSION}")
traffic_light = importlib.import_module(f"traffic_light.{TRAFFIC_VERSION}")


