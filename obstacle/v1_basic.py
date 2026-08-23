"""LiDAR 기반 장애물 회피. LaneTracker와 같은 패턴으로 상태를 인스턴스에 들고 있는
ObstacleAvoider 클래스 하나로 캡슐화했습니다 -- main()에서 한 번만 만들어서 매 프레임
step(points, curve_direction, corner_active, now)을 호출하면 됩니다.

방향 결정은 LiDAR 거리 + 차선이 아는 방향(curve_direction) 힌트로만 정합니다 (카메라 색상
기반 콘 인식은 오탐이 잦아 제거된 원본 설계를 그대로 따름). 라이다는 트랙 경계를 모르므로
"열린 공간"이 트랙 밖일 수도 있다는 전제로, 차선 방향을 기본으로 신뢰하고 확실히 막혔을 때만
override합니다.

[FIX] 원본(auto4)의 boxed_in은 전방+좌우만 보고 후방을 전혀 체크하지 않아서, 후진 탈출이
뒤쪽 장애물로 바로 후진하거나 무한 재트리거될 수 있었습니다 (코드 리뷰에서 확인). 이 모듈은
후방 여유(rear_room)를 추가로 체크하고, 후진 시도 횟수 상한(MAX_BACKUP_ATTEMPTS)을 넘으면
후진 대신 정지로 escalate합니다.
"""
import time

import config


def obstacle_steer_bias(points, curve_direction=None, prev_direction=0.0):
    """전방 라바콘/장애물로부터 회피하는 조향 보정치(deg), 정면 최소거리(m), 이번에 쓴 회피 방향,
    boxed_in(조향만으로는 못 피할 정도로 사방이 막혔는지 -- 후방까지 확인).

    curve_direction: 차선이 판단한 "이 방향이 트랙 중심/커브 안쪽"(+1=좌/-1=우) 힌트, 단서가
        없으면 None. 있으면 그 방향을 그대로 신뢰함 -- 라이다는 트랙 경계를 모르기 때문에
        "열린 공간"이 트랙 밖일 수도 있어서, 트랙 자체가 알려주는 방향이 더 안전한 우선순위임."""
    front = [(a, d) for a, d in points
             if abs(a) <= config.OBSTACLE_FRONT_HALF_ANGLE and d < config.OBSTACLE_MAX_RANGE]
    if not front:
        return 0.0, float("inf"), 0.0, False

    min_dist = min(d for _, d in front)
    intensity = min(1.0, (config.OBSTACLE_MAX_RANGE - min_dist) / config.OBSTACLE_MAX_RANGE)

    lo, hi = config.LEFT_ROOM_ANGLES
    left_room = min((d for a, d in points if lo <= a <= hi), default=config.OBSTACLE_MAX_RANGE * 4)
    lo, hi = config.RIGHT_ROOM_ANGLES
    right_room = min((d for a, d in points if lo <= a <= hi), default=config.OBSTACLE_MAX_RANGE * 4)
    room_diff = left_room - right_room

    # 후방(±180도 부근) 여유 -- 각도는 lidar()가 (-180,180]로 정규화하므로 양 끝을 함께 잡음
    rear_room = min((d for a, d in points if abs(a) >= 180 - config.REAR_HALF_ANGLE),
                     default=config.OBSTACLE_MAX_RANGE * 4)

    # 뒤도 막혀있으면 후진 자체가 위험하므로 boxed_in으로 보지 않음 -- 이 경우 main()의
    # 후진 대신-정지 경로로 넘어가야 함 (뒤로 물러날 곳이 없는데 후진을 시도하면 그대로 충돌)
    boxed_in = (min_dist < config.BACKUP_FRONT_RANGE and
                left_room < config.BACKUP_SIDE_RANGE and
                right_room < config.BACKUP_SIDE_RANGE and
                rear_room >= config.BACKUP_REAR_RANGE)

    if curve_direction is not None:
        curve_side_room = left_room if curve_direction > 0 else right_room
        other_side_room = right_room if curve_direction > 0 else left_room
        if curve_side_room < other_side_room - config.CURVE_OVERRIDE_MARGIN:
            direction = 1.0 if room_diff >= 0 else -1.0
        else:
            direction = curve_direction
    else:
        if abs(room_diff) < config.ROOM_HYSTERESIS_M and prev_direction != 0.0:
            direction = prev_direction
        else:
            direction = 1.0 if room_diff >= 0 else -1.0

    bias = direction * intensity * config.OBSTACLE_STEER_GAIN
    bias = max(-config.OBSTACLE_BIAS_MAX, min(config.OBSTACLE_BIAS_MAX, bias))
    return bias, min_dist, direction, boxed_in


class ObstacleAvoider:
    def __init__(self):
        self.obstacle_direction = 0.0
        self.smoothed_bias = 0.0
        self.last_valid_points = []
        self.last_ok_time = time.time()
        self.backing_until = 0.0     # 이 시각까지는 후진 유지 중 (0이면 후진 중 아님)
        self.backup_steer = 0.0
        self.backup_attempts = 0     # 전방이 안 뚫린 채 연속 후진한 횟수

    def step(self, points, curve_direction, corner_active, now):
        """카메라/조향 루프 한 프레임에 대응하는 회피 계산.

        반환 dict:
          bias: 조향에 더할 회피 보정치(deg, EMA 스무딩됨)
          min_dist: 정면 최소거리(m)
          backing: 지금 후진 유지 구간인지
          backup_steer / backup_speed: backing이 True일 때 쓸 조향/속도
          stopped_boxed: 후진 시도 상한을 넘겨서 후진 대신 정지해야 하는지
          lidar_stale_for: 라이다가 몇 초째 안 들어오는지
        """
        # 통신 실패/타임아웃으로 lidar()가 빈 리스트를 반환한 걸 "장애물 없음 확인됨"으로
        # 오인하지 않기 위한 TTL 처리 -- 짧은 끊김은 마지막 스캔으로 매끄럽게 넘기고,
        # 오래 끊기면 방향판단은 포기(속도만 감속)
        if points:
            self.last_valid_points = points
            self.last_ok_time = now
        stale_for = now - self.last_ok_time
        if points:
            points_for_avoidance = points
        elif stale_for <= config.LIDAR_STALE_GRACE_S:
            points_for_avoidance = self.last_valid_points
        else:
            points_for_avoidance = []

        bias, min_dist, direction, boxed_in = obstacle_steer_bias(
            points_for_avoidance, curve_direction, self.obstacle_direction)
        self.obstacle_direction = direction

        # corner_active 여부와 무관하게 계속 값을 갱신해둬야 코너를 빠져나올 때 바로 정상
        # 세기로 복귀함(0에서부터 다시 천천히 올라오지 않음)
        self.smoothed_bias = (config.OBSTACLE_BIAS_ALPHA * bias +
                               (1.0 - config.OBSTACLE_BIAS_ALPHA) * self.smoothed_bias)
        bias = self.smoothed_bias

        if corner_active and not config.CORNER_OBSTACLE_STEER and min_dist >= config.CORNER_OBSTACLE_EMERGENCY_DIST:
            # 코너 중엔 원칙적으로 조향에 반영하지 않음 -- 단, 장애물이 아주 가까워지면 예외
            bias = 0.0

        stopped_boxed = False
        if boxed_in and now >= self.backing_until:
            if self.backup_attempts >= config.MAX_BACKUP_ATTEMPTS:
                # 이미 여러 번 후진했는데도 여전히 boxed_in -- 같은 자리로 계속 후진하고
                # 있을 가능성이 높으므로 더 후진하지 않고 정지 (사람 개입 불가능한 대회 환경이라
                # "계속 시도"보다 "안전하게 멈춤"이 우선)
                stopped_boxed = True
            else:
                self.backing_until = now + config.BACKUP_DURATION_S
                self.backup_steer = config.STEER_MAX if direction >= 0 else -config.STEER_MAX
                self.backup_attempts += 1
        elif not boxed_in and now >= self.backing_until:
            # 후진 윈도우가 끝난 뒤 더 이상 boxed_in이 아니면(=전방이 뚫림) 카운터 리셋 --
            # 진짜 새 장애물을 만났을 때는 다시 처음부터 시도할 수 있어야 함
            self.backup_attempts = 0

        return {
            "bias": bias,
            "min_dist": min_dist,
            "direction": direction,
            "boxed_in": boxed_in,
            "backing": now < self.backing_until,
            "backup_steer": self.backup_steer,
            "backup_speed": config.BACKUP_SPEED,
            "backup_attempts": self.backup_attempts,
            "stopped_boxed": stopped_boxed,
            "lidar_stale_for": stale_for,
        }
