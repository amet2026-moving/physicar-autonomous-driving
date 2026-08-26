# 조향/속도 결정 튜닝값: Stanley 게인, 코너 컨트롤러, 장애물 회피 오프셋,
# 원시 충돌 감지 거리. (후진 리커버리 없음 -- utils/states.py, control/obstacle_avoid.py 참고)
#
# 실차 주행 후 다시 조정하게 될 파일. "이게 흰선/노란선/콘이 맞는가" 같은 센싱
# 임계값은 camera_params.py / lidar_params.py에 있음 -- 랩타임이 안 나온다고 색상
# 임계값부터 건드리지 않도록 이 경계를 지킬 것.

# ============================================================
# STANLEY 조향
# ============================================================
K_LATERAL = 14.0            # Stanley 횡방향 오차 게인 (무차원)
K_HEADING = 40.0             # Stanley 헤딩 오차 게인 (무차원)
STEER_ALPHA = 0.55           # 최종 조향 출력 EMA 스무딩 계수 (0~1, 클수록 최근값 반영 큼)
CORNER_STEER_ALPHA = 0.72    # 코너 활성 중 조향 EMA 스무딩 계수 (0~1)

# ============================================================
# 속도 계획
# ============================================================
SPEED_MIN = 0.28              # 커브 구간 최저 속도 (m/s)
SPEED_MAX = 0.75              # 직선 구간 최고 속도 (m/s)
SPEED_ONE_LINE = 0.34         # 차선 한쪽만 인식될 때 속도 (m/s)
YELLOW_GUIDE_SPEED = 0.28     # 흰선 약함 + 노란선 폴백 추종 중 속도 (m/s)

SPEED_RAMP_UP = 0.12          # 프레임당 최대 가속량 (m/s)
SPEED_RAMP_DOWN = 0.06        # 프레임당 최대 감속량 (m/s) -- 감속은 보수적으로 작게

CURVE_DEADBAND = 0.30         # 곡률 무시 데드밴드 (무차원, 이하는 0으로 취급)
STRAIGHT_CURVE_LIMIT = 0.18   # 직선으로 볼 곡률 상한 (무차원)
TARGET_SPEED_ALPHA = 0.45     # 목표 속도 EMA 스무딩 계수 (0~1)
PREVIEW_MIN_Y_RATIO = 0.44    # 속도계획용 전방 미리보기 샘플링 y 하한 (BEV 화면 비율, 0~1)

# ============================================================
# 코너 진입/이탈 (흰선 헤딩/조향각 기준 게이트)
# ============================================================
CORNER_ENTER_HEADING = 0.10   # 코너 진입 판정 헤딩오차 임계값 (무차원)
CORNER_ENTER_STEER = 7.0      # 코너 진입 판정 조향각 임계값 (도, deg)
CORNER_EXIT_HEADING = 0.085   # 코너 이탈(직선 복귀) 판정 헤딩오차 임계값 (무차원)
CORNER_EXIT_STEER = 6.0       # 코너 이탈 판정 조향각 임계값 (도, deg)
CORNER_EXIT_BOTH_FRAMES = 5   # 코너 이탈 확정에 필요한 연속 프레임 수 (개)

CORNER_PATH_ENTER_STEER_DEG = 4.5   # 노란 경로 기반 선제적 코너 진입 조향각 임계값 (도)
CORNER_PATH_EXIT_STEER_DEG = 3.2    # 노란 경로 기반 코너 이탈 조향각 임계값 (도)
CORNER_PATH_MIN_POINTS = 2          # 코너 경로로 인정할 최소 노란점 개수 (개)

CORNER_WEAK_WHITE_MIN_STEER_DEG = 2.8   # 흰선이 약할 때 코너진입 인정 최소 조향각 (도)
CORNER_WEAK_WHITE_CONFIRM_FRAMES = 2    # 흰선약함+코너 조건 확정에 필요한 연속 프레임 수 (개)

CORNER_OPPOSITE_SIGN_MIN_DEG = 3.0        # 반대부호 조향으로 볼 최소 각도 (도) -- 노이즈성 부호반전 방지
CORNER_OPPOSITE_SIGN_CONFIRM_FRAMES = 3   # 반대부호를 진짜 경로변경으로 인정할 연속 프레임 수 (개)

SPEED_CORNER = 0.36        # 코너 진입 직후 속도 (m/s)
SPEED_CORNER_HOLD = 0.32   # 코너 중 노란경로가 잠깐 사라졌을 때 유지 속도 (m/s)

# ============================================================
# 코너 경로 추종 (노란 점선 중앙선)
# ============================================================
# 점들을 "어떻게 연결할지"(YELLOW_PATH_*)는 인식 단계 기하값이라
# camera_params.py로 옮김. 여기 남은 CORNER_LOOKAHEAD_PX는 이미 연결된
# 경로에서 "어디를 조향 목표점으로 삼을지"를 고르는 제어 판단이라 그대로 둠.
CORNER_LOOKAHEAD_PX = 135.0          # 코너 목표점 전방 탐색 거리 (px)

CORNER_STEER_GAIN = 0.44         # 코너 목표각 -> 조향각 변환 게인 (무차원)
CORNER_MEDIUM_STEER_DEG = 8.5    # '중간 코너'로 분류할 조향각 기준 (도)
CORNER_HARD_STEER_DEG = 12.5     # '급코너'로 분류할 조향각 기준 (도)

SPEED_CORNER_GENTLE_MIN = 0.40   # 완만한 코너 최저 속도 (m/s)
SPEED_CORNER_MEDIUM_MIN = 0.36   # 중간 코너 최저 속도 (m/s)
SPEED_CORNER_MEDIUM_MAX = 0.40   # 중간 코너 최고 속도 (m/s)
SPEED_CORNER_HARD_MIN = 0.32     # 급코너 최저 속도 (m/s)
SPEED_CORNER_HARD_MAX = 0.34     # 급코너 최고 속도 (m/s)

CORNER_HOLD_FRAMES = 12   # 노란 경로가 사라져도 마지막 코너 조향을 유지할 프레임 수 (개)

REAL_CORNER_MEMORY_UPDATE_MIN_DEG = 8.0   # 코너 메모리를 갱신할 최소 조향각 (도) -- 이 이상 강한 명령만 기억
REAL_CORNER_MEMORY_RATIO = 0.68           # 코너 메모리 반영 비율 (0~1)
REAL_CORNER_MEMORY_MIN_DEG = 10.0         # 코너 메모리 클리핑 하한 (도)
REAL_CORNER_MEMORY_MAX_DEG = 14.0         # 코너 메모리 클리핑 상한 (도)
REAL_CORNER_SPEED_ALPHA = 0.78            # 코너 속도 EMA 스무딩 계수 (0~1)

REAL_CORNER_ACTIVE_MAX_SPEED = 0.50                 # 코너 활성 중 최고 속도 (m/s)
REAL_CORNER_FALLBACK_ONE_LINE_MAX_SPEED = 0.34      # 코너+한쪽 차선만 인식될 때 최고 속도 (m/s)
REAL_CORNER_FALLBACK_BOTH_MAX_SPEED = 0.40          # 코너+양쪽 차선 인식될 때 최고 속도 (m/s)

CORNER_FALLBACK_STEER_STEP_DEG = 3.0   # 코너 폴백 조향 변화량 한도 (도/프레임)
SPEED_CORNER_FALLBACK = 0.46           # 코너 폴백 속도 (m/s)

# ============================================================
# 장애물 회피 (연속함수 방식 -- control/obstacle_avoid.py 참고)
# ============================================================
AVOIDANCE_ENABLED = True   # 장애물 회피 기능 On/Off

AVOID_FULL_OFFSET_RATIO = 0.24   # 회피 시 차선폭 대비 최대 오프셋 비율 (0~1)
AVOID_OFFSET_ALPHA = 0.18        # 회피 오프셋 EMA 스무딩 계수 (0~1)
AVOID_OFFSET_DONE_RATIO = 0.018  # 오프셋이 이 비율 이하로 줄면 '복귀 완료'로 판정 (0~1)

AVOID_PASS_X_M = 0.10        # 이 거리(전후) 이내를 "바로 옆을 지나는 구간"으로 보고 회피량을 최대로 유지
AVOID_CRUISE_SPEED = 0.65    # 회피 중이 아닐 때 순항 속도 (m/s)
AVOID_EMERGENCY_SPEED = 0.18 # 회피량이 최대일 때 속도 하한 (m/s) -- 정지/후진 없음, 이 아래로는 안 내려감

# ============================================================
# 원시 충돌 감지 (콘으로 확정되지 않은 물체도 전방 근접이면 회피방향을 잡는 보강채널)
# ============================================================
# 정지/후진은 팀 결정으로 하지 않음(control/obstacle_avoid.py 참고) -- 그래서 여기엔
# "정지 거리" 같은 값이 없다. 이 물체가 콘 모양 필터(lidar_params.py)를 통과하든
# 안 하든, 아래 거리/여유 안에 들어오면 obstacle_avoid.py가 즉시 회피방향을 잡는다.
RAW_SHIELD_SOFT_HALF_WIDTH_M = 0.21       # '주의'로 볼 중심선 기준 반폭 (m)
RAW_SHIELD_BRAKE_X_M = 0.70               # 회피방향 즉시 결정을 시작할 전방거리 (m)
