# VehicleMode.OBSTACLE_AVOID용 회피 컨트롤러.
#
# "이름 붙은 단계(NORMAL/SHIFT_OUT/...)"가 아니라, 장애물까지 남은 전방거리(cx)의
# 연속함수로 회피량(0~1)을 매 프레임 바로 계산한다. 프레임간 기억은 그 값을 부드럽게
# 따라가는 EMA 하나뿐 -- 상태머신이 아니다.
#
# 회피량 곡선(사다리꼴):
#   멀리 있음(cx >= RAW_SHIELD_BRAKE_X_M)         -> 0
#   다가오는 중                                    -> 0에서 1로 증가
#   바로 옆을 지나는 구간(|cx| <= AVOID_PASS_X_M)  -> 1 유지 (실제로 부딪힐 수 있는 구간)
#   확실히 뒤로 빠짐                                -> 1에서 0으로 감소
#   완전히 지나감(cx <= -RAW_SHIELD_BRAKE_X_M)     -> 0
# "바로 옆"을 지나는 순간이 최대 회피가 필요한 지점이지, 회피가 필요없는 지점이
# 아니다 -- 대신 그 유지 구간을 짧게 잡고, 확실히 뒤로 빠지면 바로 줄인다(예전처럼
# 한참 지난 뒤까지 최대치를 끌고 가지 않는다).
#
# 팀 결정: 정지/후진은 절대 하지 않는다(부딪히는 게 낫다는 판단). 그래서 최저속도는
# AVOID_EMERGENCY_SPEED고, 0으로는 절대 안 내려간다. 회피 방향도 팀 결정으로 항상
# 오른쪽 고정(config.control_params.AVOID_DIRECTION_SIGN)이라 장애물의 실제 좌/우
# 위치는 보지 않는다 -- 콘 형상으로 아직 확정되지 않은 물체(형상필터 실패, 대각선
# 접근으로 각도창을 벗어난 경우 등)도 전방 근접 + 중심선 여유 부족이면 그대로
# 오른쪽으로 반응한다.
#
# 오프셋을 조향으로 바꾸는 계산은 control/lane_follow.steer_with_offset()을 그대로
# 재사용한다(근/원거리에 같은 오프셋을 줘서, 차선 중심 자체를 옆으로 민 것처럼
# 만든다 -- 헤딩을 인위적으로 틀지 않는 가장 단순한 방식). 튜닝 상수는
# config/control_params.py(AVOID_*)와 config/lidar_params.py(RAW_SHIELD_* 형상값)에 있음.
import numpy as np

from config import control_params as cfg
from config import lidar_params as lidar_cfg
from control import lane_follow


def _lateral_clearance(cluster):
    """차량 중심선 기준 클러스터의 가장 가까운 가장자리까지 거리(m)."""
    if cluster.y_min > 0.0:
        return cluster.y_min
    if cluster.y_max < 0.0:
        return -cluster.y_max
    return 0.0


def _is_collision_shape(cluster):
    """콘 여부와 무관하게 '뭔가 단단한 물체'로 볼만한 최소 형상 -- 라이다 노이즈
    배제용(콘 형상 필터보다 느슨함)."""
    return (
        cluster.n >= lidar_cfg.RAW_SHIELD_MIN_POINTS
        and lidar_cfg.RAW_SHIELD_MIN_WIDTH_M <= cluster.width <= lidar_cfg.RAW_SHIELD_MAX_WIDTH_M
        and cluster.angle_span <= lidar_cfg.RAW_SHIELD_MAX_ANGLE_SPAN_DEG
    )


def _closest_raw_hazard(clusters):
    """콘 형상 확정 여부와 무관하게, 전방 근접 + 중심선 여유 부족인 가장 가까운
    클러스터. classify_cone_candidates()의 각도창/모양 게이트를 벗어난 대각선
    장애물을 잡기 위한 보강 채널."""
    hits = [
        c for c in clusters
        if _is_collision_shape(c)
        and 0.0 < c.cx <= cfg.RAW_SHIELD_BRAKE_X_M
        and _lateral_clearance(c) <= cfg.RAW_SHIELD_SOFT_HALF_WIDTH_M
    ]
    if not hits:
        return None
    return min(hits, key=lambda c: c.cx)


def _avoid_magnitude(cx):
    """장애물까지 남은 전방거리(cx, m)에 따른 회피량 배율(0~1)의 사다리꼴 곡선.
    react_x보다 멀면 0, margin_x 이내(바로 옆 지나는 구간)는 1로 유지, 그보다 확실히
    뒤로 빠지면(음수 방향으로 margin_x를 넘어서면) 바로 줄어들어 react_x만큼 뒤에서
    다시 0이 된다."""
    react_x = cfg.RAW_SHIELD_BRAKE_X_M
    margin_x = cfg.AVOID_PASS_X_M
    ramp = max(react_x - margin_x, 1e-6)

    if cx >= margin_x:
        return float(np.clip((react_x - cx) / ramp, 0.0, 1.0))
    if cx >= -margin_x:
        return 1.0
    return float(np.clip((cx + react_x) / ramp, 0.0, 1.0))


class ObstacleAvoidController:
    """장애물 회피 오프셋을 프레임간에 부드럽게 이어주는 EMA 값 하나만 들고 있는
    컨트롤러(인스턴스 하나를 매 프레임 재사용). 이름 붙은 단계는 없다."""

    def __init__(self):
        self.offset_ratio = 0.0

    def reset(self):
        self.offset_ratio = 0.0

    def step(self, fusion_result, clusters, lane_obs) -> tuple[float, float, str]:
        """이번 프레임의 (조향각(도), 속도(m/s), status)를 반환.
        status는 "ACTIVE"(회피 오프셋이 아직 남아있음) / "DONE"(다음 프레임부터
        LANE_FOLLOW로 복귀해도 됨) 중 하나."""
        cluster = fusion_result.cluster or _closest_raw_hazard(clusters) if cfg.AVOIDANCE_ENABLED else None

        if cluster is not None:
            target_ratio = cfg.AVOID_DIRECTION_SIGN * _avoid_magnitude(float(cluster.cx)) * cfg.AVOID_FULL_OFFSET_RATIO
        else:
            target_ratio = 0.0

        self.offset_ratio = (
            cfg.AVOID_OFFSET_ALPHA * target_ratio + (1.0 - cfg.AVOID_OFFSET_ALPHA) * self.offset_ratio
        )

        lane_width_px = lane_obs.lane_width_px or 0.0
        offset_px = self.offset_ratio * lane_width_px
        steer = lane_follow.steer_with_offset(lane_obs, offset_px, offset_px)
        if steer is None:
            steer = 0.0   # 차선 정보 자체가 없어도 정지는 안 함 -- 직진 유지

        # 지금 얼마나 세게 피하고 있는지(0~1)에 비례해서 감속 -- AVOID_EMERGENCY_SPEED가
        # 하한이라 정지/후진은 절대 없음.
        smoothed_magnitude = float(np.clip(abs(self.offset_ratio) / cfg.AVOID_FULL_OFFSET_RATIO, 0.0, 1.0))
        speed = cfg.AVOID_CRUISE_SPEED - smoothed_magnitude * (cfg.AVOID_CRUISE_SPEED - cfg.AVOID_EMERGENCY_SPEED)

        status = "ACTIVE" if abs(self.offset_ratio) > cfg.AVOID_OFFSET_DONE_RATIO else "DONE"
        return steer, float(speed), status
