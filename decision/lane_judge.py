# 차선 원시 인식 결과(sensors/camera.py의 LaneObservation)로 LaneState를 판정하는 모듈.
#
# T_T.py 원본은 Stanley 조향각/헤딩오차(제어 게인 K_LATERAL/K_HEADING 사용)를 코너
# 판정 기준으로 썼는데, 그러면 "판단"이 "제어" 계산에 의존하게 돼서 계층이 거꾸로
# 된다. 대신 훨씬 단순하고 직관적인 기준을 쓴다: 흰선 중심선(center_near/center_far)
# 또는 노란 경로가 "근거리 대비 원거리에서 옆으로 얼마나 벌어져 있는지"(BEV px)를
# 곡률 신호로 보고, 진입/이탈 임계값을 다르게 둬서(히스테리시스) 경계에서 진동하지
# 않게 한다. 튜닝 상수는 config/decision_params.py에 있음.
from config import decision_params as cfg
from utils.states import LaneState


def _lane_curvature_px(lane_obs) -> float:
    """흰선 중심선과 노란 경로 중 더 뚜렷한 커브 신호(절대값 큰 쪽)를 px로 반환.
    둘 다 단서가 없으면 0.0(커브 아님)."""
    candidates = []

    if lane_obs.center_near is not None and lane_obs.center_far is not None:
        candidates.append(abs(lane_obs.center_far - lane_obs.center_near))

    if len(lane_obs.yellow_path) >= 2:
        candidates.append(abs(lane_obs.yellow_path[-1][0] - lane_obs.yellow_path[0][0]))

    return max(candidates, default=0.0)


def is_corner(lane_obs, corner_active_prev: bool) -> bool:
    """이번 프레임이 코너 구간으로 볼만한지 판단. 직전 프레임이 코너였는지에 따라
    다른 임계값을 써서(히스테리시스) 경계값 근처 진동을 막는다."""
    curvature = _lane_curvature_px(lane_obs)
    threshold = cfg.CORNER_EXIT_CURVATURE_PX if corner_active_prev else cfg.CORNER_ENTER_CURVATURE_PX
    return curvature >= threshold


def is_off_track(lane_obs) -> bool:
    """흰선도 노란선도 단서가 없어 주행 기준을 완전히 잃은 상태인지 판단.
    흰선이 LOST여도 노란 경로가 남아있으면 그걸로 주행 가능하므로 이탈로 보지 않는다."""
    return lane_obs.mode == "LOST" and not lane_obs.yellow_path


# judge_lane_state()가 다음 프레임 코너 히스테리시스에 쓰려고 기억해두는, 직전
# 프레임의 최종 판정 결과. is_corner()/is_off_track() 자체는 상태 없는 순수 함수로
# 남겨두고(테스트하기 쉽게), 프레임간 기억은 이 얇은 wrapper만 갖는다.
_prev_lane_state = LaneState.STRAIGHT


def judge_lane_state(lane_obs) -> LaneState:
    """이번 프레임의 LaneState(STRAIGHT/CORNER/OFF_TRACK)를 판정."""
    global _prev_lane_state

    if is_off_track(lane_obs):
        state = LaneState.OFF_TRACK
    elif is_corner(lane_obs, _prev_lane_state == LaneState.CORNER):
        state = LaneState.CORNER
    else:
        state = LaneState.STRAIGHT

    _prev_lane_state = state
    return state
