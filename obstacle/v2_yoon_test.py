"""LiDAR 기반 장애물 회피 - v2_ynz_gap_hysteresis.

이 모듈은 binned grid gap-following 알고리즘, 장애물 inflation(최소값 보존), 
동적 angular corridor 코너 벽 분리, 그리고 안전성 및 히스테리시스 상태 머신을 적용하여 
기존의 단순 거리 편차 회피 로직을 근본적으로 개선한 버전입니다.

기존 파일은 절대 수정하지 않고, 이 파일 하나로만 완전히 독립적인 drop-in 모듈로 구현되었습니다.
"""
import time
import math
import config


class ObstacleAvoider:
    def __init__(self):
        # 기존 v1_basic.py 와의 100% 인터페이스 호환을 위한 속성 유지
        self.obstacle_direction = 0.0
        self.smoothed_bias = 0.0
        self.last_valid_points = []
        self.last_ok_time = time.time()
        self.backing_until = 0.0
        self.backup_steer = 0.0
        self.backup_attempts = 0

        # 신규 안전성 및 히스테리시스 상태 머신 속성
        self.state = "NORMAL"  # NORMAL, AVOID, BACKUP, STOPPED
        self.state_entry_time = time.time()
        self.avoid_direction = 0.0  # -1.0 (우), 1.0 (좌)
        self.avoid_start_time = 0.0
        self.consecutive_normal_frames = 0
        self.consecutive_avoid_frames = 0
        self.last_gap_angle = None

        # 로컬 튜닝 및 실험용 파라미터 (명시적 출처 기술)
        self.VEHICLE_HALF_WIDTH = 0.15  # 차량 물리적 반폭 (local)
        self.SAFETY_MARGIN = 0.10       # 안전 마진 (local)
        self.REQUIRED_GAP_WIDTH = 2 * (self.VEHICLE_HALF_WIDTH + self.SAFETY_MARGIN)  # 0.50m
        self.GAP_MIN_SEARCH_DIST = 0.70  # gap 탐색 안전 하한 거리 (local)
        self.GAP_MIN_OBSERVED_RATIO = 0.50  # gap 내 observed 비율 하한
        self.GAP_MAX_CONSECUTIVE_UNKNOWN_DEG = 10.0  # gap 내 연속 unobserved 각도 상한
        
        self.AVOID_DIR_MIN_HOLD_S = 0.6  # 회피 방향 최소 유지시간
        self.MIN_SCORE_DIFF = 15.0       # 방향 전환에 필요한 최소 점수차 (300점 만점 기준 5%)
        self.AVOID_ENTER_FRAMES = 3      # 회피 진입 디바운스 프레임 수
        self.AVOID_EXIT_FRAMES = 5       # 회피 탈출 디바운스 프레임 수
        self.MAX_AVOID_TIME_S = 8.0      # 단일 회피 상태 최대 유지시간
        
        self.CORRIDOR_SHIFT_ANGLE = 15.0  # 코너 시 진행 corridor 편향 각도 (local)

    def _process_lidar(self, points):
        """LiDAR 포인트 데이터를 binned grid 및 observed mask로 변환하고 inflation을 수행합니다.
        
        grid_dist: 각도 index (0: -90도 ~ 180: +90도)에 해당하는 최소 거리
        grid_observed: 실제 유효한 센서 반사가 발생한 위치인지 나타내는 mask
        """
        grid_dist = [12.0] * 181
        grid_observed = [False] * 181

        # 1. raw LiDAR 포인트를 grid에 매핑
        for angle, d in points:
            if abs(angle) <= 90:
                idx = int(round(angle)) + 90
                # NaN, inf, rmin, rmax 유효성 필터링 (car_api.py 규격 준수)
                if d is not None and not math.isnan(d) and not math.isinf(d) and 0.02 < d < 12.0:
                    grid_dist[idx] = min(grid_dist[idx], d)
                    grid_observed[idx] = True

        # 2. 아주 작은 각도 간격(3도 이하)의 미측정(unknown) 구간을 주변 측정값으로 보간
        i = 0
        while i < 181:
            if not grid_observed[i]:
                j = i
                while j < 181 and not grid_observed[j]:
                    j += 1
                length = j - i
                # 양쪽 경계가 observed이고 구간이 3도 이하인 경우 보간
                if i > 0 and j < 181 and length <= 3:
                    d_start = grid_dist[i - 1]
                    d_end = grid_dist[j]
                    for k in range(i, j):
                        weight = (k - (i - 1)) / (length + 1)
                        grid_dist[k] = d_start + weight * (d_end - d_start)
                        grid_observed[k] = True
                i = j
            else:
                i += 1

        # 3. 최소 거리 보존 inflation 수행
        inflated_dist = list(grid_dist)
        for i in range(181):
            if grid_observed[i] and grid_dist[i] < config.OBSTACLE_MAX_RANGE:
                d = grid_dist[i]
                # 차량 안전폭(0.50m) 기준 inflation 각도 계산
                theta_inflate_rad = math.atan(self.REQUIRED_GAP_WIDTH / 2.0 / d)
                theta_inflate_deg = math.degrees(theta_inflate_rad)
                span = int(round(theta_inflate_deg))

                start_idx = max(0, i - span)
                end_idx = min(180, i + span)
                for j in range(start_idx, end_idx + 1):
                    inflated_dist[j] = min(inflated_dist[j], d)

        return grid_dist, grid_observed, inflated_dist

    def _is_wall_pattern(self, grid_dist, grid_observed, start_idx, end_idx):
        """주어진 index 구간 내 관측 데이터가 벽처럼 매끄럽고 연속적으로 연결되는지 판단합니다."""
        observed_count = 0
        smooth_transitions = 0
        prev_d = None

        for idx in range(start_idx, end_idx + 1):
            if grid_observed[idx]:
                observed_count += 1
                d = grid_dist[idx]
                if prev_d is not None:
                    if abs(d - prev_d) <= 0.15:  # 인접 픽셀간 편차가 15cm 이내일 때
                        smooth_transitions += 1
                prev_d = d
            else:
                prev_d = None

        total_bins = end_idx - start_idx + 1
        if total_bins >= 5 and observed_count >= total_bins * 0.6:
            if smooth_transitions >= observed_count * 0.7:
                return True
        return False

    def _apply_suppression(self, grid_dist, grid_observed, inflated_dist, curve_direction, corner_active):
        """코너링 시 진행 경로 corridor 밖의 연속된 코너 벽 형상에 대해서만 가중치(suppression)를 적용합니다."""
        adjusted_dist = list(inflated_dist)
        if not corner_active or curve_direction is None:
            return adjusted_dist

        C = curve_direction
        theta_corridor = C * self.CORRIDOR_SHIFT_ANGLE

        for i in range(181):
            d = inflated_dist[i]
            if d >= config.OBSTACLE_MAX_RANGE:
                continue

            angle = i - 90
            angle_rot = angle - theta_corridor
            angle_rot_rad = math.radians(angle_rot)

            # coordinate projection
            x_rot = d * math.sin(angle_rot_rad)
            y_rot = d * math.cos(angle_rot_rad)

            # 동적 angular corridor 내부 여부 확인
            in_corridor = (abs(x_rot) <= (self.VEHICLE_HALF_WIDTH + self.SAFETY_MARGIN) and 
                           0.0 <= y_rot <= config.OBSTACLE_MAX_RANGE)

            if in_corridor:
                w = 1.0  # 실제 진행 경로 내의 장애물은 절대 억제 안 함
            else:
                # 15도 window 영역 내의 코너 벽 패턴 검사
                start_win = max(0, i - 7)
                end_win = min(180, i + 7)
                is_wall = self._is_wall_pattern(grid_dist, grid_observed, start_win, end_win)

                if is_wall:
                    if x_rot * C < 0:
                        # 외측 벽 (suppression 대폭 적용)
                        w = 0.2
                    else:
                        # 내측 벽
                        if abs(x_rot) <= 0.40:
                            w = 1.0  # 너무 가깝게 붙은 내측 벽은 충돌 예방을 위해 억제하지 않음
                        else:
                            w = 0.5  # 먼 내측 벽은 오탐 방지용 부분 억제
                else:
                    w = 1.0  # corridor 밖이라도 단독 장애물(라바콘 등)은 억제하지 않음

            # 위협 거리를 억제 가중치에 맞춰 조절
            adjusted_dist[i] = config.OBSTACLE_MAX_RANGE - w * (config.OBSTACLE_MAX_RANGE - d)

        return adjusted_dist

    def _find_gaps(self, adjusted_dist, grid_observed, d_safe):
        """주어진 safe distance 기준으로 차량이 통과할 수 있는 유효 gap들을 탐색합니다."""
        gaps = []
        in_gap = False
        start_idx = -1

        for i in range(181):
            if adjusted_dist[i] >= d_safe:
                if not in_gap:
                    in_gap = True
                    start_idx = i
            else:
                if in_gap:
                    in_gap = False
                    gaps.append((start_idx, i - 1))
        if in_gap:
            gaps.append((start_idx, 180))

        valid_gaps = []
        for s, e in gaps:
            d_min = min(adjusted_dist[s:e+1])
            width_deg = e - s + 1

            # Chord length (물리 폭) 계산
            width_rad = math.radians(width_deg)
            w_physical = 2 * d_min * math.sin(width_rad / 2.0)

            # 1. 물리적 최소 gap 폭 검증 (0.50m 미만 거부)
            if w_physical < self.REQUIRED_GAP_WIDTH:
                continue

            # 2. observed coverage 비율 검증
            gap_observed_bins = grid_observed[s:e+1]
            observed_ratio = sum(gap_observed_bins) / len(gap_observed_bins)
            if observed_ratio < self.GAP_MIN_OBSERVED_RATIO:
                continue

            # 3. 최대 연속 unknown 각도 검증 (센서 사각지대 진입 방지)
            max_consec_unknown = 0
            curr_consec = 0
            for obs in gap_observed_bins:
                if not obs:
                    curr_consec += 1
                    max_consec_unknown = max(max_consec_unknown, curr_consec)
                else:
                    curr_consec = 0
            if max_consec_unknown > self.GAP_MAX_CONSECUTIVE_UNKNOWN_DEG:
                continue

            valid_gaps.append({
                "start": s,
                "end": e,
                "width_deg": width_deg,
                "d_min": d_min,
                "center_angle": (s + e) / 2.0 - 90.0,
                # 거리는 최대 OBSTACLE_MAX_RANGE로 상한을 제한하여, max_range 12.0m가 점수를 과부풀리지 않도록 함
                "d_mean_capped": sum(min(d, config.OBSTACLE_MAX_RANGE) for d in adjusted_dist[s:e+1]) / (e - s + 1)
            })

        return valid_gaps

    def _score_gaps(self, gaps, curve_direction):
        """정규화된 각 항목별 계산식을 이용해 gap의 우선순위 점수를 산출합니다."""
        scored_gaps = []

        # curve_direction 방향에 따라 목표 조향 각 설정 (+:좌 / -:우)
        if curve_direction == 1.0:
            theta_target = 30.0
        elif curve_direction == -1.0:
            theta_target = -30.0
        else:
            theta_target = 0.0

        for gap in gaps:
            # Width Score (최대 180도 기준 정규화, [0, 100])
            s_width = 100.0 * (gap["width_deg"] / 180.0)

            # Depth Score (최대 OBSTACLE_MAX_RANGE 기준 정규화, [0, 100])
            s_depth = 100.0 * (gap["d_mean_capped"] / config.OBSTACLE_MAX_RANGE)

            # Alignment Penalty (최대 180도 편차 기준 정규화, [0, 100])
            deviation = abs(gap["center_angle"] - theta_target)
            p_alignment = 100.0 * (deviation / 180.0)

            # Hysteresis Bonus (이전 gap 조향 유지 보너스, [0, 20])
            b_hysteresis = 0.0
            if self.last_gap_angle is not None:
                if abs(gap["center_angle"] - self.last_gap_angle) <= 20.0:
                    b_hysteresis = 20.0

            # Raw score 합계 (범위: [-100, 220])
            raw_score = s_width + s_depth - p_alignment + b_hysteresis
            
            # 최종 점수를 양수로 변환하고 [0, 300] 범위로 명시적 clamp
            score = max(0.0, min(300.0, raw_score + 100.0))

            gap["score"] = score
            scored_gaps.append(gap)

        # 점수 내림차순 정렬
        scored_gaps.sort(key=lambda x: x["score"], reverse=True)
        return scored_gaps

    def step(self, points, curve_direction, corner_active, now):
        """메인 제어 루프의 매 프레임마다 장애물 회피 조향을 계산합니다.
        
        반환값: main.py의 ObstacleAvoider 규격과 동일한 dict 구조
        """
        # 1. LiDAR TTL 및 데이터 상태 유효성 검사 (안전성)
        if points:
            self.last_valid_points = points
            self.last_ok_time = now
            
        stale_for = now - self.last_ok_time

        # LiDAR 데이터 수신이 0.5초(LIDAR_STALE_GRACE_S)를 넘는 경우 즉시 안전 정지 요청
        if stale_for > config.LIDAR_STALE_GRACE_S:
            if stale_for > 2.0:
                print(f"[CRITICAL] LiDAR stale for {stale_for:.1f}s -- backup timeout exceeded.")
            return {
                "bias": 0.0,
                "min_dist": float("inf"),
                "direction": 0.0,
                "boxed_in": False,
                "backing": False,
                "backup_steer": 0.0,
                "backup_speed": config.BACKUP_SPEED,
                "backup_attempts": self.backup_attempts,
                "stopped_boxed": True,  # main.py에서 steer=0, speed=0 으로 안전 정지
                "lidar_stale_for": stale_for,
            }

        points_for_avoidance = points if points else self.last_valid_points

        # 2. 기존 v1_basic.py 와 100% 호환되는 raw 거리 검사 및 후진 판단 로직
        front = [(a, d) for a, d in points_for_avoidance
                 if abs(a) <= config.OBSTACLE_FRONT_HALF_ANGLE and d < config.OBSTACLE_MAX_RANGE]
        min_dist = min(d for _, d in front) if front else float("inf")

        lo, hi = config.LEFT_ROOM_ANGLES
        left_room = min((d for a, d in points_for_avoidance if lo <= a <= hi), default=config.OBSTACLE_MAX_RANGE * 4)
        lo, hi = config.RIGHT_ROOM_ANGLES
        right_room = min((d for a, d in points_for_avoidance if lo <= a <= hi), default=config.OBSTACLE_MAX_RANGE * 4)
        rear_room = min((d for a, d in points_for_avoidance if abs(a) >= 180 - config.REAR_HALF_ANGLE),
                         default=config.OBSTACLE_MAX_RANGE * 4)

        # front arc 내의 LiDAR 포인트 개수가 극도로 부족한 경우 (Blind Spot) 세이프 정지 요청
        front_points_count = sum(1 for a, _ in points_for_avoidance if abs(a) <= 90)
        if front_points_count < 5:
            return {
                "bias": 0.0,
                "min_dist": min_dist,
                "direction": 0.0,
                "boxed_in": False,
                "backing": False,
                "backup_steer": 0.0,
                "backup_speed": config.BACKUP_SPEED,
                "backup_attempts": self.backup_attempts,
                "stopped_boxed": True,
                "lidar_stale_for": stale_for,
            }

        # v1_basic.py 와 완벽히 동일한 조건의 boxed_in 판정
        boxed_in = (min_dist < config.BACKUP_FRONT_RANGE and
                    left_room < config.BACKUP_SIDE_RANGE and
                    right_room < config.BACKUP_SIDE_RANGE and
                    rear_room >= config.BACKUP_REAR_RANGE)

        # 3. 현재 후진 중인 경우의 상태 유지 처리
        if now < self.backing_until:
            return {
                "bias": 0.0,
                "min_dist": min_dist,
                "direction": self.obstacle_direction,
                "boxed_in": boxed_in,
                "backing": True,
                "backup_steer": self.backup_steer,
                "backup_speed": config.BACKUP_SPEED,
                "backup_attempts": self.backup_attempts,
                "stopped_boxed": False,
                "lidar_stale_for": stale_for,
            }

        # 후진이 끝난 후 전방이 확보되었으면 시도 횟수 리셋
        if not boxed_in:
            self.backup_attempts = 0

        # 4. Hysteresis 상태 머신 전이 판정
        has_obstacle = (min_dist < config.OBSTACLE_MAX_RANGE)
        
        if has_obstacle:
            self.consecutive_avoid_frames += 1
            self.consecutive_normal_frames = 0
        else:
            self.consecutive_normal_frames += 1
            self.consecutive_avoid_frames = 0

        # NORMAL -> AVOID 전이 (디바운스 반영)
        if self.state == "NORMAL" and self.consecutive_avoid_frames >= self.AVOID_ENTER_FRAMES:
            self.state = "AVOID"
            self.state_entry_time = now
            self.avoid_start_time = now

        # AVOID -> NORMAL 전이 (디바운스 반영)
        if self.state == "AVOID" and self.consecutive_normal_frames >= self.AVOID_EXIT_FRAMES:
            self.state = "NORMAL"
            self.state_entry_time = now
            self.last_gap_angle = None
            self.avoid_direction = 0.0

        # AVOID 상태 단일 유지시간 초과 시의 안전 정지/후진 에스컬레이션
        if self.state == "AVOID" and (now - self.avoid_start_time > self.MAX_AVOID_TIME_S):
            if rear_room >= config.BACKUP_REAR_RANGE and self.backup_attempts < config.MAX_BACKUP_ATTEMPTS:
                # 후방 개방 시 후진 상태로 비상 탈출 시도
                self.state = "BACKUP"
                self.state_entry_time = now
                self.backing_until = now + config.BACKUP_DURATION_S
                self.backup_steer = config.STEER_MAX if self.obstacle_direction >= 0 else -config.STEER_MAX
                self.backup_attempts += 1
                return {
                    "bias": 0.0,
                    "min_dist": min_dist,
                    "direction": self.obstacle_direction,
                    "boxed_in": boxed_in,
                    "backing": True,
                    "backup_steer": self.backup_steer,
                    "backup_speed": config.BACKUP_SPEED,
                    "backup_attempts": self.backup_attempts,
                    "stopped_boxed": False,
                    "lidar_stale_for": stale_for,
                }
            else:
                self.state = "STOPPED"
                self.state_entry_time = now

        # boxed_in 상태에 의한 즉시 후진/정지 전이
        if boxed_in:
            if self.backup_attempts >= config.MAX_BACKUP_ATTEMPTS:
                self.state = "STOPPED"
            else:
                self.state = "BACKUP"
                self.state_entry_time = now
                self.backing_until = now + config.BACKUP_DURATION_S
                self.backup_steer = config.STEER_MAX if self.obstacle_direction >= 0 else -config.STEER_MAX
                self.backup_attempts += 1
                return {
                    "bias": 0.0,
                    "min_dist": min_dist,
                    "direction": self.obstacle_direction,
                    "boxed_in": boxed_in,
                    "backing": True,
                    "backup_steer": self.backup_steer,
                    "backup_speed": config.BACKUP_SPEED,
                    "backup_attempts": self.backup_attempts,
                    "stopped_boxed": False,
                    "lidar_stale_for": stale_for,
                }

        # STOPPED 상태 도달 시 즉각 정지 명령 반환
        if self.state == "STOPPED":
            return {
                "bias": 0.0,
                "min_dist": min_dist,
                "direction": self.obstacle_direction,
                "boxed_in": boxed_in,
                "backing": False,
                "backup_steer": 0.0,
                "backup_speed": config.BACKUP_SPEED,
                "backup_attempts": self.backup_attempts,
                "stopped_boxed": True,
                "lidar_stale_for": stale_for,
            }

        # 5. AVOID 상태 조향 계산 (Binned Gap-Following)
        target_bias = 0.0
        if self.state == "AVOID":
            grid_dist, grid_observed, inflated_dist = self._process_lidar(points_for_avoidance)
            adjusted_dist = self._apply_suppression(grid_dist, grid_observed, inflated_dist, curve_direction, corner_active)

            # 다단계 D_safe 탐색 진행
            scored_gaps = []
            d_safe = config.OBSTACLE_MAX_RANGE
            while d_safe >= self.GAP_MIN_SEARCH_DIST:
                gaps = self._find_gaps(adjusted_dist, grid_observed, d_safe)
                if gaps:
                    scored_gaps = self._score_gaps(gaps, curve_direction)
                    break
                d_safe -= 0.1

            # 유효 gap을 발견한 경우
            if scored_gaps:
                selected_gap = scored_gaps[0]

                # 6. 방향 유지 록(Direction Hold) 및 전환 마진 적용
                if self.last_gap_angle is not None and self.avoid_direction != 0.0:
                    # 방향 고정 시간 이내인 경우 기존과 동일한 조향 부호 우선
                    if now - self.avoid_start_time < self.AVOID_DIR_MIN_HOLD_S:
                        same_side_gaps = [g for g in scored_gaps if g["center_angle"] * self.avoid_direction >= 0.0]
                        if same_side_gaps:
                            selected_gap = same_side_gaps[0]
                    else:
                        # 방향 전환이 가능한 시간이나, 점수 차이가 MIN_SCORE_DIFF를 만족해야만 전환 허용
                        best_gap = scored_gaps[0]
                        best_gap_dir = 1.0 if best_gap["center_angle"] >= 0.0 else -1.0
                        if best_gap_dir != self.avoid_direction:
                            old_side_gaps = [g for g in scored_gaps if g["center_angle"] * self.avoid_direction >= 0.0]
                            old_best_score = old_side_gaps[0]["score"] if old_side_gaps else 0.0
                            if best_gap["score"] >= old_best_score + self.MIN_SCORE_DIFF:
                                selected_gap = best_gap
                                self.avoid_start_time = now
                            elif old_side_gaps:
                                selected_gap = old_side_gaps[0]
                else:
                    self.avoid_start_time = now

                self.last_gap_angle = selected_gap["center_angle"]
                self.obstacle_direction = 1.0 if selected_gap["center_angle"] >= 0.0 else -1.0
                self.avoid_direction = self.obstacle_direction

                # 조향 편향값(bias) 계산 및 맵핑
                intensity = min(1.0, (config.OBSTACLE_MAX_RANGE - min_dist) / config.OBSTACLE_MAX_RANGE)
                target_bias = selected_gap["center_angle"] * intensity * (config.OBSTACLE_STEER_GAIN / 30.0)
                target_bias = max(-config.OBSTACLE_BIAS_MAX, min(config.OBSTACLE_BIAS_MAX, target_bias))
            else:
                # 장애물이 감지되었으나 지나갈 수 있는 유효 gap이 없는 경우 정지 Escalation
                self.obstacle_direction = 0.0
                self.state = "STOPPED"
                return {
                    "bias": 0.0,
                    "min_dist": min_dist,
                    "direction": 0.0,
                    "boxed_in": True,
                    "backing": False,
                    "backup_steer": 0.0,
                    "backup_speed": config.BACKUP_SPEED,
                    "backup_attempts": self.backup_attempts,
                    "stopped_boxed": True,
                    "lidar_stale_for": stale_for,
                }

        # Smoothed steering bias 적용
        self.smoothed_bias = (config.OBSTACLE_BIAS_ALPHA * target_bias + 
                              (1.0 - config.OBSTACLE_BIAS_ALPHA) * self.smoothed_bias)

        return {
            "bias": self.smoothed_bias,
            "min_dist": min_dist,
            "direction": self.obstacle_direction,
            "boxed_in": boxed_in,
            "backing": False,
            "backup_steer": 0.0,
            "backup_speed": config.BACKUP_SPEED,
            "backup_attempts": self.backup_attempts,
            "stopped_boxed": False,
            "lidar_stale_for": stale_for,
        }