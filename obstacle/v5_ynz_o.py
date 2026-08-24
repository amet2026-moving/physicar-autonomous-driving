"""LiDAR 기반 장애물 회피 - v5_ynz_disparity_extender.

이 모듈은 CPSWeek 2019 F1TENTH 우승 알고리즘인 Disparity Extender와 
조향 블렌딩(Steering Blending)을 기존 상태 머신에 융합한 "버전 5" 개량형 코드입니다.

핵심 기능:
1. Disparity Extender (거리 편차 확장):
   LiDAR 빔 간의 급격한 거리 편차(disparity)를 감지하고, 차량 반폭 + 안전 마진(0.70m) 크기의 
   가상 버블을 장애물 에지에 씌워 제외함으로써 장애물 모서리 충돌을 원천 차단합니다.
2. 조향 블렌딩 (Steering Blending):
   lane 제어기(LaneKeeper)의 조향값과 obstacle 제어기의 조향값을 장애물 근접도(w)에 따라 
   선형적으로 부드럽게 합성하여 급격한 조향 변동을 차단하고 안정성을 극대화합니다.
   Steer_blended = (1 - w) * Steer_lane + w * Steer_obstacle
3. 동적 차선 변경(Left/Right Lane) 트리거 유지.
"""
import time
import math
import config


class ObstacleAvoider:
    def __init__(self):
        # 기존 main.py 및 v1_basic.py 와의 100% 인터페이스 호환을 위한 속성 유지
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

        # v5 파라미터 (Disparity Extender + Blending)
        self.VEHICLE_HALF_WIDTH = 0.15  # 차량 물리적 반폭
        self.SAFETY_MARGIN = 0.20       # 안전 여유폭 0.20m 확장 (합산 0.70m gap)
        self.REQUIRED_GAP_WIDTH = 2 * (self.VEHICLE_HALF_WIDTH + self.SAFETY_MARGIN)  # 0.70m
        self.GAP_MIN_SEARCH_DIST = 0.70  # gap 탐색 안전 하한 거리
        self.GAP_MIN_OBSERVED_RATIO = 0.50  # gap 내 observed 비율 하한
        self.GAP_MAX_CONSECUTIVE_UNKNOWN_DEG = 10.0  # gap 내 연속 unobserved 각도 상한
        
        self.AVOID_DIR_MIN_HOLD_S = 0.6  # 회피 방향 최소 유지시간
        self.MIN_SCORE_DIFF = 15.0       # 방향 전환에 필요한 최소 점수차
        self.AVOID_ENTER_FRAMES = 1      # 장애물 감지 시 즉각 진입
        self.AVOID_EXIT_FRAMES = 5       # 회피 탈출 디바운스 프레임 수
        self.MAX_AVOID_TIME_S = 8.0      # 단일 회피 상태 최대 유지시간
        
        self.CORRIDOR_SHIFT_ANGLE = 15.0  # 코너 시 진행 corridor 편향 각도
        
        self.OBSTACLE_BIAS_ALPHA = 0.9    # 조향 반응 지연 최소화
        self.SLOWDOWN_EXTRA_MARGIN = 0.40 # 감속 시점을 당기기 위한 보정값 (m)
        self.LEFT_GAP_BONUS = 40.0        # 왼쪽 gap 선택 기본 보너스
        self.last_obstacle_time = 0.0     # 마지막으로 좌측 장애물을 목격한 시각

        # 초기 타겟 차선을 명시적으로 설정
        config.target_lane = "LEFT"

    def _process_lidar_disparity(self, points):
        """LiDAR 포인트 데이터에 Disparity Extender 알고리즘을 적용하여 빈 영역을 확장합니다."""
        grid_dist = [12.0] * 181
        grid_observed = [False] * 181

        # 1. raw LiDAR 포인트를 grid에 매핑
        for angle, d in points:
            if abs(angle) <= 90:
                idx = int(round(angle)) + 90
                if d is not None and not math.isnan(d) and not math.isinf(d) and 0.02 < d < 12.0:
                    grid_dist[idx] = min(grid_dist[idx], d)
                    grid_observed[idx] = True

        # 2. 아주 작은 각도 간격(3도 이하) 보간
        i = 0
        while i < 181:
            if not grid_observed[i]:
                j = i
                while j < 181 and not grid_observed[j]:
                    j += 1
                length = j - i
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

        # 3. Disparity Extender: 연속된 두 빔 사이의 거리 격차(Disparity)를 감지하고 확장
        # 거리 차가 0.4m 이상인 경계를 감지하여 가까운 쪽 거리값으로 먼 쪽을 덮어씀 (버블 씌우기)
        disparity_dist = list(grid_dist)
        
        for idx in range(1, 180):
            d_curr = grid_dist[idx]
            d_next = grid_dist[idx + 1]
            
            if grid_observed[idx] and grid_observed[idx + 1]:
                diff = d_next - d_curr
                if abs(diff) >= 0.40:
                    # 격차가 큰 장애물 경계를 찾음. 가까운 쪽 거리를 사용해 버블 확장 각도 산출
                    d_near = min(d_curr, d_next)
                    theta_bubble_rad = math.atan(self.REQUIRED_GAP_WIDTH / 2.0 / d_near)
                    theta_bubble_deg = math.degrees(theta_bubble_rad)
                    span = int(round(theta_bubble_deg))
                    
                    if diff > 0:
                        # 오른쪽에서 왼쪽으로 멀어짐 -> index + 1 방향으로 덮어씀
                        start_idx = idx + 1
                        end_idx = min(180, idx + span)
                        for j in range(start_idx, end_idx + 1):
                            disparity_dist[j] = min(disparity_dist[j], d_near)
                    else:
                        # 왼쪽에서 오른쪽으로 멀어짐 -> index 방향(아래)으로 덮어씀
                        start_idx = max(0, idx - span + 1)
                        end_idx = idx
                        for j in range(start_idx, end_idx + 1):
                            disparity_dist[j] = min(disparity_dist[j], d_near)

        # 4. 최소 거리 보존 inflation 병합 적용
        inflated_dist = list(disparity_dist)
        for i in range(181):
            if grid_observed[i] and grid_dist[i] < config.OBSTACLE_MAX_RANGE:
                d = grid_dist[i]
                theta_inflate_rad = math.atan(self.REQUIRED_GAP_WIDTH / 2.0 / d)
                theta_inflate_deg = math.degrees(theta_inflate_rad)
                span = int(round(theta_inflate_deg))

                start_idx = max(0, i - span)
                end_idx = min(180, i + span)
                for j in range(start_idx, end_idx + 1):
                    inflated_dist[j] = min(inflated_dist[j], d)

        return grid_dist, grid_observed, inflated_dist

    def _is_wall_pattern(self, grid_dist, grid_observed, start_idx, end_idx):
        observed_count = 0
        smooth_transitions = 0
        prev_d = None

        for idx in range(start_idx, end_idx + 1):
            if grid_observed[idx]:
                observed_count += 1
                d = grid_dist[idx]
                if prev_d is not None:
                    if abs(d - prev_d) <= 0.15:
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

            x_rot = d * math.sin(angle_rot_rad)
            y_rot = d * math.cos(angle_rot_rad)

            in_corridor = (abs(x_rot) <= (self.VEHICLE_HALF_WIDTH + self.SAFETY_MARGIN) and 
                           0.0 <= y_rot <= config.OBSTACLE_MAX_RANGE)

            if in_corridor:
                w = 1.0
            else:
                start_win = max(0, i - 7)
                end_win = min(180, i + 7)
                is_wall = self._is_wall_pattern(grid_dist, grid_observed, start_win, end_win)

                if is_wall:
                    if x_rot * C < 0:
                        w = 0.2
                    else:
                        if abs(x_rot) <= 0.40:
                            w = 1.0
                        else:
                            w = 0.5
                else:
                    w = 1.0

            adjusted_dist[i] = config.OBSTACLE_MAX_RANGE - w * (config.OBSTACLE_MAX_RANGE - d)

        return adjusted_dist

    def _find_gaps(self, adjusted_dist, grid_observed, d_safe):
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

            width_rad = math.radians(width_deg)
            w_physical = 2 * d_min * math.sin(width_rad / 2.0)

            if w_physical < self.REQUIRED_GAP_WIDTH:
                continue

            gap_observed_bins = grid_observed[s:e+1]
            observed_ratio = sum(gap_observed_bins) / len(gap_observed_bins)
            if observed_ratio < self.GAP_MIN_OBSERVED_RATIO:
                continue

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
                "d_mean_capped": sum(min(d, config.OBSTACLE_MAX_RANGE) for d in adjusted_dist[s:e+1]) / (e - s + 1)
            })

        return valid_gaps

    def _score_gaps(self, gaps, curve_direction):
        scored_gaps = []

        if curve_direction == 1.0:
            theta_target = 35.0
        elif curve_direction == -1.0:
            theta_target = -30.0
        else:
            theta_target = 25.0

        for gap in gaps:
            s_width = 100.0 * (gap["width_deg"] / 180.0)
            s_depth = 100.0 * (gap["d_mean_capped"] / config.OBSTACLE_MAX_RANGE)

            deviation = abs(gap["center_angle"] - theta_target)
            p_alignment = 50.0 * (deviation / 180.0)

            b_hysteresis = 0.0
            if self.last_gap_angle is not None:
                if abs(gap["center_angle"] - self.last_gap_angle) <= 20.0:
                    b_hysteresis = 20.0

            left_bonus = self.LEFT_GAP_BONUS if gap["center_angle"] >= 0.0 else 0.0

            raw_score = s_width + s_depth - p_alignment + b_hysteresis + left_bonus
            score = max(0.0, min(300.0, raw_score + 100.0))

            gap["score"] = score
            scored_gaps.append(gap)

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

        # LiDAR 데이터 수신이 0.5초를 넘는 경우 즉시 안전 정지 요청
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
                "stopped_boxed": True,
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

        front_points_count = sum(1 for a, _ in points_for_avoidance if abs(a) <= 90)
        
        # 기동 시간 고려하여 스타트 초기 0.15초 이내면 스킵
        if front_points_count < 5 and stale_for > 0.15:
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

        # 4. 왼쪽 차선(Left Lane) 장애물 감지에 따른 동적 차선 스위칭 제어
        left_lane_obstacle = any(d < config.OBSTACLE_MAX_RANGE and -20 <= a <= 45 for a, d in points_for_avoidance)
        
        if left_lane_obstacle:
            config.target_lane = "RIGHT"
            self.last_obstacle_time = now
        else:
            if now - self.last_obstacle_time > 2.0:
                config.target_lane = "LEFT"

        right_obstacle = any(d < config.OBSTACLE_MAX_RANGE and -90 <= a <= -10 for a, d in points_for_avoidance)
        should_avoid = has_obstacle = (min_dist < config.OBSTACLE_MAX_RANGE) or right_obstacle
        
        if should_avoid:
            self.consecutive_avoid_frames += 1
            self.consecutive_normal_frames = 0
        else:
            self.consecutive_normal_frames += 1
            self.consecutive_avoid_frames = 0

        # NORMAL -> AVOID 전이 (장애물 발견 시 1프레임만에 즉시 진입)
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

        # 5. AVOID 상태 조향 계산 (Binned Gap-Following with Disparity Extender)
        steer_obstacle = 0.0
        if self.state == "AVOID":
            grid_dist, grid_observed, inflated_dist = self._process_lidar_disparity(points_for_avoidance)
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
                    if now - self.avoid_start_time < self.AVOID_DIR_MIN_HOLD_S:
                        same_side_gaps = [g for g in scored_gaps if g["center_angle"] * self.avoid_direction >= 0.0]
                        if same_side_gaps:
                            selected_gap = same_side_gaps[0]
                    else:
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

                # 조향 이탈 게인을 높여서 (config.OBSTACLE_STEER_GAIN / 20.0) 큰 장애물 피하기 각도 확보
                steer_obstacle = selected_gap["center_angle"] * (config.OBSTACLE_STEER_GAIN / 20.0)
                
                # 최대 조향각 한계 2배 증폭
                obstacle_bias_max = 2.0 * config.STEER_MAX  # 40.0 도
                steer_obstacle = max(-obstacle_bias_max, min(obstacle_bias_max, steer_obstacle))
            else:
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

        # 6. 조향 블렌딩 (Steering Blending) 적용
        # keeper의 실시간 smoothed_steer 를 획득
        keeper = getattr(config, "keeper_instance", None)
        steer_lane = keeper.smoothed_steer if keeper else 0.0

        # 장애물 거리 기반 합성 가중치(w) 계산
        # 1.90m 이상 -> w = 0.0 (차선 전전) / 0.70m 이하 -> w = 1.0 (회피 전전)
        if min_dist >= config.OBSTACLE_MAX_RANGE:
            w_blend = 0.0
        elif min_dist <= self.GAP_MIN_SEARCH_DIST:
            w_blend = 1.0
        else:
            w_blend = (config.OBSTACLE_MAX_RANGE - min_dist) / (config.OBSTACLE_MAX_RANGE - self.GAP_MIN_SEARCH_DIST)

        # 블렌딩 조향 계산: Steer_blended = (1 - w) * Steer_lane + w * Steer_obstacle
        steer_blended = (1.0 - w_blend) * steer_lane + w_blend * steer_obstacle
        steer_blended = max(-config.STEER_MAX, min(config.STEER_MAX, steer_blended))

        # main.py의 Steer_final = Steer_lane + Bias 관계식에 부합하는 Bias 계산
        target_bias = steer_blended - steer_lane

        # Smoothed steering bias 적용
        self.smoothed_bias = (self.OBSTACLE_BIAS_ALPHA * target_bias + 
                              (1.0 - self.OBSTACLE_BIAS_ALPHA) * self.smoothed_bias)

        # 감속 시작 시점 가속을 위한 min_dist 보정
        returned_min_dist = max(0.1, min_dist - self.SLOWDOWN_EXTRA_MARGIN) if should_avoid else min_dist

        return {
            "bias": self.smoothed_bias,
            "min_dist": returned_min_dist,
            "direction": self.obstacle_direction,
            "boxed_in": boxed_in,
            "backing": False,
            "backup_steer": 0.0,
            "backup_speed": config.BACKUP_SPEED,
            "backup_attempts": self.backup_attempts,
            "stopped_boxed": False,
            "lidar_stale_for": stale_for,
        }