# VehicleMode.OBSTACLE_AVOID용 회피 컨트롤러.
#
# T_T.py의 ConeAvoidanceFSM(5단계: NORMAL -> SHIFT_OUT -> PASS -> SHIFT_IN -> RECOVER
# -> NORMAL)을 포팅. 5단계는 이 클래스 안의 구현 디테일이고, control/modes.py에는
# step()이 반환하는 "ACTIVE"/"DONE" 상태만 보인다. RECOVER는 "차선 중앙으로 복귀"
# 단계일 뿐, 후진 리커버리와는 무관하다(이 프로젝트엔 그 모드 자체가 없음).
#
# 팀 결정: 정지/후진은 절대 하지 않는다(트랙에서 멈추거나 뒤로 가는 것보다 부딪히는
# 게 낫다는 판단). 그래서 원본에 있던 "원시 충돌 쉴드"의 정지 반응은 전부 없앴다.
# 대신, 콘 형상으로 아직 확정되지 않은 물체(형상필터 실패, 대각선 접근으로 각도창을
# 벗어난 경우 등)도 전방 근접 + 중심선 여유 부족이면 raw 클러스터의 cy 부호만으로
# 즉시 임시 회피방향을 잡아 SHIFT_OUT에 진입한다 -- 원본은 이 경우 브레이크만 걸고
# 회피 방향은 못 정했었음(대각선 장애물 앞에서 서버리는 원인이었던 부분).
#
# 오프셋을 조향으로 바꾸는 계산(steering_from_lane_with_offsets에 대응)은
# control/lane_follow.steer_with_offset()을 그대로 재사용한다. 튜닝 상수는
# config/control_params.py(AVOID_*)와 config/lidar_params.py(RAW_SHIELD_* 형상값)에
# 있음.
import numpy as np

from config import control_params as cfg
from config import lidar_params as lidar_cfg
from control import lane_follow
from utils.states import ObstacleState


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


def _fallback_sign(cluster):
    """콘으로 확정 안 된 raw 클러스터의 cy 부호로 즉석 회피방향 결정.
    cy>0(좌측에 있음) -> 오른쪽으로 피함(+1), cy<=0(우측) -> 왼쪽으로 피함(-1)."""
    return +1.0 if float(cluster.cy) > 0.0 else -1.0


class ObstacleAvoidController:
    """장애물 회피 상태를 프레임간에 들고 있는 컨트롤러 (인스턴스 하나를 매 프레임 재사용)."""

    def __init__(self):
        self.state = "NORMAL"
        self.offset_sign = 0.0
        self.near_offset_ratio = 0.0
        self.far_offset_ratio = 0.0
        self.latched_lane_width_px = None
        self.last_cluster = None

    def reset(self):
        self.__init__()

    def _latch_lane_width(self, lane_width_px):
        if lane_width_px is not None and np.isfinite(lane_width_px) and lane_width_px > 50.0:
            self.latched_lane_width_px = float(lane_width_px)

    def _enter_shift_out(self, sign, lane_obs):
        self.offset_sign = sign
        self._latch_lane_width(lane_obs.lane_width_px)
        self.state = "SHIFT_OUT"

    def _pick_sign(self, obstacle_state, cluster):
        if obstacle_state == ObstacleState.LEFT:
            return +1.0
        if obstacle_state == ObstacleState.RIGHT:
            return -1.0
        if cluster is not None:
            return _fallback_sign(cluster)
        return None

    def _update_state(self, obstacle_state, fusion_result, clusters, lane_obs):
        cluster = fusion_result.cluster or _closest_raw_hazard(clusters)
        if cluster is not None:
            self.last_cluster = cluster

        close_enough = cluster is not None and 0.0 < cluster.cx <= cfg.RAW_SHIELD_BRAKE_X_M
        detected = obstacle_state != ObstacleState.CLEAR or close_enough

        # NORMAL에서 새로 진입하는 것과, 복귀 중(SHIFT_IN/RECOVER) 새 장애물이 확인돼
        # 복귀를 중단하고 다시 회피를 시작하는 것을 같은 트리거로 처리한다.
        if detected and self.state in ("NORMAL", "SHIFT_IN", "RECOVER"):
            sign = self._pick_sign(obstacle_state, cluster)
            if sign is not None:
                self._enter_shift_out(sign, lane_obs)
                return

        if self.state == "SHIFT_OUT" and cluster is not None:
            clearance = _lateral_clearance(cluster)
            clearance_pass = (
                cluster.cx <= cfg.AVOID_PASS_CLEARANCE_MAX_X_M
                and clearance >= cfg.AVOID_PASS_CLEARANCE_M
            )
            if cluster.cx <= cfg.AVOID_PASS_X_M or clearance_pass:
                self.state = "PASS"

        elif self.state == "PASS" and (cluster is None or cluster.cx < -0.03):
            self.state = "SHIFT_IN"

        # SHIFT_IN -> RECOVER -> NORMAL 전이는 오프셋 크기로 _step_offsets()에서 판단.

    def _target_offsets(self):
        s = self.offset_sign
        if self.state == "SHIFT_OUT":
            return s * cfg.AVOID_SHIFT_NEAR_RATIO, s * cfg.AVOID_FULL_OFFSET_RATIO
        if self.state == "PASS":
            return s * cfg.AVOID_FULL_OFFSET_RATIO, s * cfg.AVOID_FULL_OFFSET_RATIO
        if self.state == "SHIFT_IN":
            return s * cfg.AVOID_FULL_OFFSET_RATIO, 0.0
        return 0.0, 0.0   # NORMAL, RECOVER

    def _step_offsets(self):
        target_near, target_far = self._target_offsets()
        self.near_offset_ratio = (
            cfg.AVOID_OFFSET_ALPHA * target_near + (1.0 - cfg.AVOID_OFFSET_ALPHA) * self.near_offset_ratio
        )
        self.far_offset_ratio = (
            cfg.AVOID_OFFSET_ALPHA * target_far + (1.0 - cfg.AVOID_OFFSET_ALPHA) * self.far_offset_ratio
        )

        if self.state == "SHIFT_IN" and abs(self.far_offset_ratio) <= 0.035:
            self.state = "RECOVER"
        elif self.state == "RECOVER" and (
            abs(self.near_offset_ratio) <= cfg.AVOID_OFFSET_DONE_RATIO
            and abs(self.far_offset_ratio) <= cfg.AVOID_OFFSET_DONE_RATIO
        ):
            near, far = self.near_offset_ratio, self.far_offset_ratio
            self.reset()
            return near, far

        return self.near_offset_ratio, self.far_offset_ratio

    def _speed_cap(self):
        if self.state == "SHIFT_OUT":
            cap = cfg.AVOID_SHIFT_SPEED
        elif self.state == "PASS":
            cap = cfg.AVOID_PASS_SPEED
        elif self.state == "SHIFT_IN":
            cap = cfg.AVOID_RETURN_SPEED
        elif self.state == "RECOVER":
            cap = cfg.AVOID_RECOVER_SPEED
        else:
            return None

        cluster = self.last_cluster
        if cluster is not None:
            cx, cy = float(cluster.cx), abs(float(cluster.cy))
            if 0.0 < cx <= cfg.AVOID_LATE_X_M:
                cap = min(cap, cfg.AVOID_LATE_SPEED)
            # AVOID_EMERGENCY_SPEED가 이 컨트롤러의 최저 속도다 -- 정지/후진은 절대
            # 하지 않는다는 팀 결정에 따라, 여기서도 0으로는 절대 떨어지지 않는다.
            if 0.0 < cx <= cfg.AVOID_EMERGENCY_X_M and cy <= cfg.AVOID_EMERGENCY_LATERAL_M:
                cap = min(cap, cfg.AVOID_EMERGENCY_SPEED)

        return cap

    def step(self, obstacle_state, fusion_result, clusters, lane_obs) -> tuple[float, float, str]:
        """이번 프레임의 (조향각(도), 속도(m/s), status)를 반환.
        status는 "ACTIVE"(회피 진행중) / "DONE"(회피 완료, 다음 프레임부터
        LANE_FOLLOW로 복귀) 중 하나."""
        self._update_state(obstacle_state, fusion_result, clusters, lane_obs)
        near_ratio, far_ratio = self._step_offsets()

        lane_width_px = self.latched_lane_width_px or lane_obs.lane_width_px or 0.0
        steer = lane_follow.steer_with_offset(
            lane_obs, near_ratio * lane_width_px, far_ratio * lane_width_px,
        )
        if steer is None:
            steer = 0.0   # 차선 정보 자체가 없어도 정지는 안 함 -- 직진 유지

        cap = self._speed_cap()
        speed = cfg.AVOID_CRUISE_SPEED if cap is None else cap

        status = "ACTIVE" if self.state != "NORMAL" else "DONE"
        return steer, speed, status
