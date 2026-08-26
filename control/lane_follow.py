# VehicleMode.LANE_FOLLOW용 Stanley 조향 + 코너 경로추종 + 속도계획 컨트롤러.
#
# LaneState(STRAIGHT/CORNER/OFF_TRACK) 판정은 decision/lane_judge.py가 이미 끝내놓은
# 것을 받기만 한다 -- 여기서 다시 코너 진입/이탈을 판단하지 않는다. Recovery 모드가
# 없으므로(정지/후진 없음 -- 팀 결정) OFF_TRACK도 여기서 "마지막 조향을 유지한 채
# 서행"으로 직접 처리한다.
#
# T_T.py의 steering_from_lane()/steering_from_lane_with_offsets()/
# steering_from_corner_target()/choose_corner_target()/choose_target_speed()/
# ramp_speed()를 그대로 포팅했다. steer_with_offset()은 control/obstacle_avoid.py가
# 회피 오프셋을 더해 같은 Stanley 계산을 재사용하기 위한 공개 진입점.
# 튜닝 상수는 config/control_params.py에 있음.
import math

import numpy as np

from config import base as base_cfg
from config import control_params as cfg
from utils.states import LaneState


def _stanley_terms(lane_obs, near_offset_px=0.0, far_offset_px=0.0):
    """Stanley 횡오차/헤딩오차 계산 (차선 중심에 오프셋을 더한 뒤). 차선 중심 정보가
    없으면 (None, None)."""
    if lane_obs.center_near is None or lane_obs.center_far is None or not lane_obs.bev_w:
        return None, None

    half_w = lane_obs.bev_w / 2.0
    target_near = lane_obs.center_near + near_offset_px
    target_far = lane_obs.center_far + far_offset_px

    lateral_error = (target_near - half_w) / half_w
    heading_error = (target_far - target_near) / half_w
    return lateral_error, heading_error


def _stanley_steer(lane_obs):
    lateral_error, heading_error = _stanley_terms(lane_obs)
    if lateral_error is None:
        return None, None

    steer = -(cfg.K_LATERAL * lateral_error + cfg.K_HEADING * heading_error)
    return float(np.clip(steer, -base_cfg.STEER_MAX, base_cfg.STEER_MAX)), heading_error


def steer_with_offset(lane_obs, near_offset_px, far_offset_px):
    """control/obstacle_avoid.py 전용 공개 진입점: 차선추종과 같은 Stanley 계산에
    목표 중심점만 좌우로 옮겨서(오프셋) 회피 조향을 만든다. 차선 정보가 없으면 None."""
    lateral_error, heading_error = _stanley_terms(lane_obs, near_offset_px, far_offset_px)
    if lateral_error is None:
        return None

    steer = -(cfg.K_LATERAL * lateral_error + cfg.K_HEADING * heading_error)
    return float(np.clip(steer, -base_cfg.STEER_MAX, base_cfg.STEER_MAX))


def _corner_target(lane_obs):
    """노란 경로 위 전방주시(lookahead) 목표점 -- 차량 위치(BEV 하단 중앙)에서
    CORNER_LOOKAHEAD_PX만큼 누적거리 이동한 지점. 노란 경로 점이
    CORNER_PATH_MIN_POINTS 미만이면 None -- 코너에 막 진입해 점 1개짜리 경로만
    보일 때는 그 점 하나의 좌우 위치가 노이즈 수준이라, 방향을 잘못 잡아 순간적으로
    반대쪽(예: 왼쪽 코너인데 오른쪽)으로 튀는 원인이 된다."""
    path = lane_obs.yellow_path
    if len(path) < cfg.CORNER_PATH_MIN_POINTS:
        return None

    w, h = lane_obs.bev_w, lane_obs.bev_h
    prev = np.array([w / 2.0, h - 8.0], dtype=np.float32)
    travelled = 0.0
    target = np.array(path[-1], dtype=np.float32)

    for p in path:
        q = np.array(p, dtype=np.float32)
        travelled += float(np.linalg.norm(q - prev))
        target = q
        if travelled >= cfg.CORNER_LOOKAHEAD_PX:
            break
        prev = q

    return float(target[0]), float(target[1])


def _corner_steer(lane_obs):
    """목표점 방향각 -> 조향각(도). 목표점이 없으면 None."""
    target = _corner_target(lane_obs)
    if target is None:
        return None

    w, h = lane_obs.bev_w, lane_obs.bev_h
    vehicle_x, vehicle_y = w / 2.0, h - 8.0
    dx = target[0] - vehicle_x
    # 급코너에서 목표점이 거의 옆으로 가도 atan2가 잘 정의되게 최소 전방거리를 둔다.
    forward = max(12.0, vehicle_y - target[1])

    target_angle_deg = math.degrees(math.atan2(dx, forward))
    steer = -cfg.CORNER_STEER_GAIN * target_angle_deg   # +steer = 좌회전
    return float(np.clip(steer, -base_cfg.STEER_MAX, base_cfg.STEER_MAX))


def _straight_speed(lane_obs, steer, heading_error):
    """곡률 기반 속도 계획(직선/한쪽차선 폴백 공용). steer/heading_error가 없으면
    (차선 정보 없음) 최소 속도로 서행."""
    if lane_obs.mode == "LOST":
        return cfg.SPEED_MIN
    if lane_obs.mode in ("LEFT_ONLY", "YELLOW_ONLY"):
        return cfg.SPEED_ONE_LINE
    if steer is None or heading_error is None:
        return cfg.SPEED_MIN

    curvature = max(
        float(np.clip(abs(heading_error) / 0.22, 0.0, 1.0)),
        float(np.clip(abs(steer) / 18.0, 0.0, 1.0)),
    )
    if curvature <= cfg.STRAIGHT_CURVE_LIMIT:
        return cfg.SPEED_MAX

    effective = np.clip(
        (curvature - cfg.CURVE_DEADBAND) / max(1.0 - cfg.CURVE_DEADBAND, 1e-6),
        0.0, 1.0,
    )
    return float(np.clip(
        cfg.SPEED_MAX - effective * (cfg.SPEED_MAX - cfg.SPEED_MIN),
        cfg.SPEED_MIN, cfg.SPEED_MAX,
    ))


def _corner_speed(base_speed, steer_deg):
    """직선용으로 계산한 속도를, 코너 조향각 크기(완만/중간/급)에 맞는 구간으로
    한 번 더 clip한다."""
    mag = abs(steer_deg)
    if mag >= cfg.CORNER_HARD_STEER_DEG:
        lo, hi = cfg.SPEED_CORNER_HARD_MIN, cfg.SPEED_CORNER_HARD_MAX
    elif mag >= cfg.CORNER_MEDIUM_STEER_DEG:
        lo, hi = cfg.SPEED_CORNER_MEDIUM_MIN, cfg.SPEED_CORNER_MEDIUM_MAX
    else:
        lo, hi = cfg.SPEED_CORNER_GENTLE_MIN, cfg.REAL_CORNER_ACTIVE_MAX_SPEED
    return float(np.clip(base_speed, lo, hi))


def _ramp_speed(current, target):
    if target > current:
        return min(target, current + cfg.SPEED_RAMP_UP)
    return max(target, current - cfg.SPEED_RAMP_DOWN)


class _LaneFollowController:
    """코너 메모리/조향 EMA처럼 프레임간에 기억해야 하는 상태를 들고 있는 내부
    컨트롤러. sensors/camera.py의 _lane_tracker 싱글턴과 같은 패턴 -- compute()는
    상태 없는 함수 모양을 유지하면서 실제 상태는 여기(모듈 전역 싱글턴)에 둔다."""

    def __init__(self):
        self.last_corner_steering = 0.0
        self.corner_hold_count = 0
        self.corner_memory_steering = 0.0
        self.smoothed_steer = 0.0

    def compute(self, lane_obs, lane_state, current_speed):
        stanley_steer, heading_error = _stanley_steer(lane_obs)

        if lane_state == LaneState.CORNER:
            raw_steer, target_speed, alpha = self._corner_branch(
                lane_obs, stanley_steer, heading_error,
            )
        else:
            self.corner_hold_count = 0
            self.corner_memory_steering = 0.0
            alpha = cfg.STEER_ALPHA

            if stanley_steer is not None:
                raw_steer = stanley_steer
                self.last_corner_steering = stanley_steer
                target_speed = _straight_speed(lane_obs, stanley_steer, heading_error)
            else:
                # OFF_TRACK -- 정지/후진 없이 마지막 조향을 유지한 채 서행한다.
                raw_steer = self.smoothed_steer
                target_speed = cfg.SPEED_MIN

        self.smoothed_steer = alpha * raw_steer + (1.0 - alpha) * self.smoothed_steer
        steer = float(np.clip(self.smoothed_steer, -base_cfg.STEER_MAX, base_cfg.STEER_MAX))
        speed = _ramp_speed(current_speed, target_speed)
        return steer, speed

    def _corner_branch(self, lane_obs, stanley_steer, heading_error):
        corner_steer = _corner_steer(lane_obs)
        alpha = cfg.CORNER_STEER_ALPHA

        if corner_steer is not None:
            self.last_corner_steering = corner_steer
            self.corner_hold_count = 0
            # 노이즈성 미세 조향은 코너 메모리에 남기지 않는다 -- 진짜 강한 명령만 기억.
            if abs(corner_steer) >= cfg.REAL_CORNER_MEMORY_UPDATE_MIN_DEG:
                self.corner_memory_steering = corner_steer
            target_speed = _corner_speed(
                _straight_speed(lane_obs, corner_steer, heading_error), corner_steer,
            )
            return corner_steer, target_speed, alpha

        if self.corner_hold_count < cfg.CORNER_HOLD_FRAMES:
            # 노란 경로가 잠깐 끊김 -- 직전 코너 조향을 그대로 유지.
            self.corner_hold_count += 1
            return self.last_corner_steering, cfg.SPEED_CORNER_HOLD, alpha

        if stanley_steer is not None and lane_obs.mode != "LOST":
            # 코너 유지시간도 끝남 -- Stanley 쪽으로 프레임당 최대 각도만큼만 복귀.
            delta = float(np.clip(
                stanley_steer - self.last_corner_steering,
                -cfg.CORNER_FALLBACK_STEER_STEP_DEG,
                cfg.CORNER_FALLBACK_STEER_STEP_DEG,
            ))
            raw_steer = self.last_corner_steering + delta

            if lane_obs.mode in ("LEFT_ONLY", "YELLOW_ONLY") and abs(self.corner_memory_steering) > 1e-3:
                # 한쪽 차선만 보이는 동안은, 아직 안 끝난 급코너를 조기에 풀지
                # 않도록 코너 메모리로 최소 크기(floor)를 보장한다.
                floor_mag = float(np.clip(
                    abs(self.corner_memory_steering) * cfg.REAL_CORNER_MEMORY_RATIO,
                    cfg.REAL_CORNER_MEMORY_MIN_DEG, cfg.REAL_CORNER_MEMORY_MAX_DEG,
                ))
                if self.corner_memory_steering > 0:
                    raw_steer = max(raw_steer, floor_mag)
                else:
                    raw_steer = min(raw_steer, -floor_mag)

            raw_steer = float(np.clip(raw_steer, -base_cfg.STEER_MAX, base_cfg.STEER_MAX))
            self.last_corner_steering = raw_steer
            target_speed = _straight_speed(lane_obs, stanley_steer, heading_error)
            return raw_steer, target_speed, alpha

        # 차선 정보 자체가 없음(사실상 OFF_TRACK) -- 마지막 조향 유지, 서행.
        return self.last_corner_steering, cfg.SPEED_MIN, alpha


_controller = _LaneFollowController()


def compute(lane_obs, lane_state, current_speed: float) -> tuple[float, float]:
    """차선 인식 결과 + 현재 속도로 (조향각(도), 속도(m/s))를 계산해서 반환."""
    return _controller.compute(lane_obs, lane_state, current_speed)


def sync_steer(steer: float) -> None:
    """VehicleMode.OBSTACLE_AVOID 중에는 compute()가 아예 호출되지 않아 조향 EMA가
    회피 시작 전 값에 멈춰있게 된다 -- 회피가 끝나고 복귀하는 순간 그 오래된 값에서부터
    다시 스무딩을 시작하면 잠깐 어긋난다. main.py가 회피 중에도 매 프레임 이걸 불러
    EMA가 실제 조향을 계속 따라가게 해서, 복귀가 이어지게 한다."""
    _controller.smoothed_steer = steer
    _controller.last_corner_steering = steer
