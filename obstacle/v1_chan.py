"""LiDAR 기반 장애물 회피 -- 인공 포텐셜필드(artificial potential field) 방식.

[출처] 팀 저장소 밖의 개인 실험 폴더(code_v3/drive_potential_field.py)의 obstacle_field_bias()를
그대로 이식. 원본은 자전거모델 기반 합성 시나리오(code_v3/lidar_param_sweep.py)로 회귀 테스트는
했지만, 이 저장소의 실제 로봇/트랙에서는 아직 한 번도 실주행 검증이 안 된 상태.

[방식] 다른 버전들(v2_gap_cluster/v2_yoon_test 등)처럼 "이 틈/저 방향 중 하나를 고른다"는
이산적 판단이 없음 -- 시야 안의 라이다 포인트 하나하나를 독립적인 반발력(1/거리에 비례) 벡터로
취급해서 전부 합산하고, 그 좌우 성분을 그대로 조향 보정치로 씀. 장애물이 정확히 대칭(가운데)이면
반발력이 상쇄되는 포텐셜필드의 잘 알려진 한계가 있어서, 그런 경우에만(front_blocked 판정)
거리 비례로 세지는 강제 회피(FIELD_TIEBREAK_*)로 자연 반발력을 완전히 대체함.

[각도 부호] car_api.lidar()가 문서화한 "+=좌측" 그대로 사용(반전 없음) -- 원본이 자기 실차
테스트로 반전 로직을 오히려 제거하며 확인한 값. *** 주의: 이 저장소의 obstacle/v2_gap_cluster.py는
정반대로 "이 로봇은 반전이 필요하다"고 실차로 확인했다고 적혀 있어 두 문서의 결론이 서로 다름 --
이 버전을 실제로 켜서 테스트할 때 가장 먼저 확인할 것(콘이 오른쪽에 있는데 오른쪽으로 꺾이면
즉시 중단하고 부호 반전 필요). ***

[main.py 미반영] 원본 main()은 아래 CORNER_SUPPRESS_*(코너 중 연속 억제)는 그대로 갖고 있지만,
그 외에 line_steer와 부호가 반대일 때 blend 비중을 조절하는 로직(EMERGENCY_DIST,
NORMAL_OPPOSE_MIN_WEIGHT, _opposing_blend)도 같이 쓰고 있었음. 그건 line_steer 값 자체가
필요해서 step(points, curve_direction, corner_active, now) 인터페이스만으로는 계산할 수 없음
-- v2_gap_cluster.py가 감속/긴급블렌딩을 미반영이라고 적어둔 것과 같은 이유로 여기서도
제외했음. 채택해서 실제로 쓰려면 이 블렌딩을 main.py 쪽에 추가하는 걸 팀 논의 후 고려할 것.

[후진 로직 없음] 원본 설계 자체가 후진 없이 근접 시 속도만 낮추는 방식 -- 이 클래스도 다른
후진-미사용 버전들과 동일하게 boxed_in/backing을 항상 False로 반환함(인터페이스 유지 목적).

*** FIELD_* 상수 전부 -- code_v3에서 2026-08-23~24 실측(일부는 합성 시뮬레이션) 기반이지만,
이 저장소(실제 트랙/이 차량)에서는 미검증. 채택 전 재튜닝 필요. ***
"""
import math
import time

import numpy as np

import config

FIELD_MIN_DIST = 0.10                       # 반발력 공식의 1/d 항이 너무 커지는 것 방지용 거리 하한(m)
FIELD_REPULSE_GAIN = 6.0                    # 자연 반발력 전역 게인
MAX_SWERVE_BIAS_DEG = 8.0                   # 최종 조향 보정치 상한(deg)
FIELD_TARGET_ALPHA = 0.5                    # bias EMA 스무딩 계수
FIELD_MIN_LATERAL = 0.5                     # 이보다 자연 반발력이 작으면 "대칭" 취급
FIELD_TIEBREAK_TRIGGER_DIST = 1.30          # 이 거리(m)부터 강제 회피 트리거 후보
FIELD_TIEBREAK_RELEASE_DIST = 1.50          # 강제 회피 활성 유지(히스테리시스) 상한 거리
FIELD_TIEBREAK_MIN_DEG = 4.0                # 강제 회피 시작 세기
FIELD_TIEBREAK_MAX_DEG = 7.0                # 강제 회피 최대 세기(BOXED_DIST 근접 시)
FIELD_TIEBREAK_FRONT_HALF_ANGLE_DEG = 5.0   # 강제 회피 "새로 트리거" 판정용 좁은 정면 각도
FIELD_TIEBREAK_CENTER_BAND_DEG = 18.0       # 강제 회피 "유지" 판정용 넓은 정면 각도
BOXED_DIST = 0.18                           # 강제 회피 세기가 최댓값에 도달하는 근접 거리

CORNER_SUPPRESS_RANGE = 1.20        # 코너 중 억제 가중치가 0에서 커지기 시작하는 거리
CORNER_SUPPRESS_FULL_DIST = 0.35    # 이 거리 이하면 코너 중에도 회피 100% 반영


def obstacle_field_bias(points, tiebreak_state, prev_bias=0.0, curve_direction=None):
    """전방 라이다 포인트를 반발력 벡터로 합산해 조향 보정치(deg)를 계산.

    tiebreak_state: {"dir": +1.0/-1.0/None, "active": bool} -- 강제 회피 방향/활성 상태를
        프레임 간 유지하기 위한 상태(호출자가 만들어서 매 프레임 그대로 넘겨야 함).
    prev_bias: 직전 프레임의(EMA 스무딩된) bias -- 강제 회피 방향을 처음 정할 때 자연
        반발력이 거의 0이면 이 부호를 대신 씀.
    curve_direction: 차선이 아는 트랙 안쪽 방향(+1=좌/-1=우) 힌트.

    반환: (bias_deg, min_dist, debug)"""
    front = [(a, d) for a, d in points
             if abs(a) <= config.OBSTACLE_FRONT_HALF_ANGLE and d < config.OBSTACLE_MAX_RANGE]
    if not front:
        tiebreak_state["dir"] = None
        tiebreak_state["active"] = False
        return 0.0, math.inf, {"lateral": 0.0, "center_blocked": False,
                                "front_blocked": False, "active": False}

    min_dist = min(d for _, d in front)

    lateral = 0.0
    for a, d in front:
        dist = max(d, FIELD_MIN_DIST)
        magnitude = FIELD_REPULSE_GAIN * (1.0 / dist - 1.0 / config.OBSTACLE_MAX_RANGE)
        lateral += -magnitude * math.sin(math.radians(a))

    natural_lateral = lateral  # 강제 개입 전 값 -- 디버그 로그용으로 보존

    center_blocked = any(d < FIELD_TIEBREAK_TRIGGER_DIST for a, d in front
                          if abs(a) <= FIELD_TIEBREAK_CENTER_BAND_DEG)
    front_blocked = any(d < FIELD_TIEBREAK_TRIGGER_DIST for a, d in front
                         if abs(a) <= FIELD_TIEBREAK_FRONT_HALF_ANGLE_DEG)

    newly_triggered = (front_blocked and abs(lateral) < FIELD_MIN_LATERAL and
                        min_dist < FIELD_TIEBREAK_TRIGGER_DIST)
    stay_active = (tiebreak_state.get("active", False) and center_blocked and
                   min_dist < FIELD_TIEBREAK_RELEASE_DIST)

    if newly_triggered or stay_active:
        if tiebreak_state["dir"] is None:
            if abs(natural_lateral) > 1e-3:
                tiebreak_state["dir"] = 1.0 if natural_lateral > 0.0 else -1.0
            elif abs(prev_bias) > FIELD_MIN_LATERAL:
                tiebreak_state["dir"] = 1.0 if prev_bias > 0.0 else -1.0
            elif curve_direction is not None:
                tiebreak_state["dir"] = curve_direction
            else:
                tiebreak_state["dir"] = 1.0 if prev_bias >= 0.0 else -1.0
        elif (abs(natural_lateral) > FIELD_MIN_LATERAL and
              (natural_lateral > 0.0) != (tiebreak_state["dir"] > 0.0)):
            tiebreak_state["dir"] = 1.0 if natural_lateral > 0.0 else -1.0
        tiebreak_state["active"] = True
        urgency = float(np.clip(
            (FIELD_TIEBREAK_TRIGGER_DIST - min_dist) /
            max(FIELD_TIEBREAK_TRIGGER_DIST - BOXED_DIST, 1e-6), 0.0, 1.0))
        forced = FIELD_TIEBREAK_MIN_DEG + urgency * (FIELD_TIEBREAK_MAX_DEG - FIELD_TIEBREAK_MIN_DEG)
        lateral = tiebreak_state["dir"] * forced
    else:
        tiebreak_state["dir"] = None
        tiebreak_state["active"] = False

    bias = float(np.clip(lateral, -MAX_SWERVE_BIAS_DEG, MAX_SWERVE_BIAS_DEG))
    debug = {"lateral": natural_lateral, "center_blocked": center_blocked,
             "front_blocked": front_blocked, "active": tiebreak_state["active"]}
    return bias, min_dist, debug


class ObstacleAvoider:
    """v1_basic.ObstacleAvoider와 동일한 인터페이스(step(points, curve_direction,
    corner_active, now) -> dict) -- main.py 수정 없이 versions.py의 OBSTACLE_VERSION만
    바꿔서 바로 교체할 수 있습니다."""

    def __init__(self):
        self.tiebreak_state = {"dir": None, "active": False}
        self.smoothed_bias = 0.0
        self.last_valid_points = []
        self.last_ok_time = time.time()

    def step(self, points, curve_direction, corner_active, now):
        # 통신 실패/타임아웃으로 lidar()가 빈 리스트를 반환한 걸 "장애물 없음 확인됨"으로
        # 오인하지 않기 위한 TTL 처리 (다른 버전들과 동일)
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

        raw_bias, min_dist, field_debug = obstacle_field_bias(
            points_for_avoidance, self.tiebreak_state, self.smoothed_bias, curve_direction)

        # EMA 스무딩은 원본(code_v3 main())과 같은 순서로: 스무딩 -> (필요시) 코너 억제.
        # 다음 프레임 tiebreak 방향 판단에 쓰이는 self.smoothed_bias는 코너 억제 이전 값을
        # 그대로 유지함(원본에서도 corner_weight는 별도 지역변수에만 곱함).
        self.smoothed_bias = (FIELD_TARGET_ALPHA * raw_bias +
                               (1.0 - FIELD_TARGET_ALPHA) * self.smoothed_bias)
        bias = self.smoothed_bias

        if corner_active:
            # 코너 중엔 on/off 스위치 대신, 장애물이 가까워질수록 0->1로 매끄럽게 커지는
            # 연속 가중치를 곱함(원본 main()의 corner_weight 그대로 이식). line_steer가
            # 필요한 반대부호 블렌딩(EMERGENCY_DIST 이하)은 이 인터페이스로 못 가져오므로
            # 미포함 -- 위 docstring 참고.
            corner_weight = max(0.0, min(1.0, (CORNER_SUPPRESS_RANGE - min_dist) /
                                 max(CORNER_SUPPRESS_RANGE - CORNER_SUPPRESS_FULL_DIST, 1e-6)))
            bias *= corner_weight

        return {
            "bias": bias,
            "min_dist": min_dist,
            "direction": 1.0 if bias >= 0 else -1.0,
            "boxed_in": False,
            "backing": False,
            "backup_steer": 0.0,
            "backup_speed": config.BACKUP_SPEED,
            "backup_attempts": 0,
            "stopped_boxed": False,
            "lidar_stale_for": stale_for,
            "field_debug": field_debug,   # 디버그/웹뷰 표시용, main.py는 안 읽음
        }
