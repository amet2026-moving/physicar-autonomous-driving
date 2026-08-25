# VehicleMode.LANE_FOLLOW용 Stanley 조향 + 속도계획 컨트롤러.
#
# LaneState.STRAIGHT와 CORNER 둘 다 여기서 처리(내부 분기) -- FSM은 어느 내부 분기를
# 탈지만 알려줄 뿐, CORNER가 별도 VehicleMode를 갖지는 않음.
#
# TODO: T_T.py에서 아래를 포팅할 것
#   - steering_from_lane() / steering_from_lane_with_offsets() (직선 구간)
#   - steering_from_corner_target() / choose_corner_target() (코너 구간)
#   - choose_target_speed() / ramp_speed() (속도 계획, 두 구간 공통)
# 튜닝 상수는 config/control_params.py에 있음.


def compute(lane_obs, lane_state, current_speed: float) -> tuple[float, float]:
    """차선 인식 결과 + 현재 속도로 (조향각(도), 속도(m/s))를 계산해서 반환. TODO."""
    raise NotImplementedError
