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

# 직선 구간에서 mode=="YELLOW_ONLY"(왼쪽 흰선이 카메라 시야를 벗어나 노란선만 보임)일
# 때, 코리도 폭(lane_obs.lane_width_px) 대비 목표 중심점을 얼마나 오른쪽으로 미는지.
# 왼쪽 흰선이 사라지는 건 대개 차가 코리도 중심보다 너무 왼쪽에 붙어서라, 살짝
# 오른쪽으로 붙여 흰선이 다시 시야에 들어오게 유도한다. 회피용 AVOID_FULL_OFFSET_RATIO
# (1.05, 반대 차로까지 건너감)보다 훨씬 작게 잡음 -- 이건 차로 변경이 아니라 미세
# 보정이므로. 실차 확인 후 재조정할 것.
YELLOW_ONLY_SEARCH_OFFSET_RATIO = 0.20

# ============================================================
# 속도 계획
# ============================================================
# 아래 속도 상수들은 전반적으로 체감 속도가 느리다는 실차 피드백에 따라 전부
# 20% 일괄 인상했다(가속/감속 램프인 SPEED_RAMP_UP/DOWN과 하드웨어 한계인
# STEER_MAX는 "속도"가 아니므로 대상에서 제외).
SPEED_MIN = 0.36              # 커브 구간 최저 속도 (m/s)
# 직선 구간 최고 속도 (m/s). 0.75 -> 0.9로 올렸던 것(실차 요청, 2.0은 과해서 보류)을
# 이번에 다시 20% 인상. 빨라진 만큼 장애물까지 도달하는 시간이 짧아지므로, 회피
# 반응거리/EMA 속도도 같이 올려뒀다(RAW_SHIELD_BRAKE_X_M, AVOID_OFFSET_ALPHA,
# decision_params.OBSTACLE_REACT_RANGE_M 참고) -- 속도만 올리면 회피가 더 늦어
# 보이는 역효과가 남.
SPEED_MAX = 1.08
SPEED_ONE_LINE = 0.48         # 차선 한쪽만 인식될 때 속도 (m/s)
YELLOW_GUIDE_SPEED = 0.34     # 흰선 약함 + 노란선 폴백 추종 중 속도 (m/s)

SPEED_RAMP_UP = 0.20          # 프레임당 최대 가속량 (m/s)
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

SPEED_CORNER = 0.43        # 코너 진입 직후 속도 (m/s)
SPEED_CORNER_HOLD = 0.34   # 코너 중 노란경로가 잠깐 사라졌을 때 유지 속도 (m/s)

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

SPEED_CORNER_GENTLE_MIN = 0.48   # 완만한 코너 최저 속도 (m/s)
SPEED_CORNER_MEDIUM_MIN = 0.43   # 중간 코너 최저 속도 (m/s)
SPEED_CORNER_MEDIUM_MAX = 0.48   # 중간 코너 최고 속도 (m/s)
SPEED_CORNER_HARD_MIN = 0.38     # 급코너 최저 속도 (m/s)
SPEED_CORNER_HARD_MAX = 0.41     # 급코너 최고 속도 (m/s)

CORNER_HOLD_FRAMES = 12   # 노란 경로가 사라져도 마지막 코너 조향을 유지할 프레임 수 (개)

REAL_CORNER_MEMORY_UPDATE_MIN_DEG = 8.0   # 코너 메모리를 갱신할 최소 조향각 (도) -- 이 이상 강한 명령만 기억
REAL_CORNER_MEMORY_RATIO = 0.68           # 코너 메모리 반영 비율 (0~1)
REAL_CORNER_MEMORY_MIN_DEG = 10.0         # 코너 메모리 클리핑 하한 (도)
REAL_CORNER_MEMORY_MAX_DEG = 14.0         # 코너 메모리 클리핑 상한 (도)
REAL_CORNER_SPEED_ALPHA = 0.78            # 코너 속도 EMA 스무딩 계수 (0~1)

REAL_CORNER_ACTIVE_MAX_SPEED = 0.60                 # 코너 활성 중 최고 속도 (m/s)
REAL_CORNER_FALLBACK_ONE_LINE_MAX_SPEED = 0.41      # 코너+한쪽 차선만 인식될 때 최고 속도 (m/s)
REAL_CORNER_FALLBACK_BOTH_MAX_SPEED = 0.48          # 코너+양쪽 차선 인식될 때 최고 속도 (m/s)

CORNER_FALLBACK_STEER_STEP_DEG = 3.0   # 코너 폴백 조향 변화량 한도 (도/프레임)
SPEED_CORNER_FALLBACK = 0.55           # 코너 폴백 속도 (m/s)

# ============================================================
# 장애물 회피 (연속함수 방식 -- control/obstacle_avoid.py 참고)
# ============================================================
AVOIDANCE_ENABLED = True   # 장애물 회피 기능 On/Off

# 회피 방향 부호 고정: +1=오른쪽. 팀 결정("항상 오른쪽으로 피했다가 돌아온다")에 따라
# 장애물의 좌/우 위치와 무관하게 이 값 하나로 방향을 정한다.
AVOID_DIRECTION_SIGN = 1.0

# 회피 시 코리도 폭(lane_obs.lane_width_px, 왼쪽흰선~노란선) 대비 최대 오프셋 비율.
# 왼쪽 차로 중앙에서 노란선 건너 오른쪽 차로 중앙까지 이동하려면 코리도 폭만큼(=1.0)
# 옆으로 빠져야 하므로 예전(좌우 흰선 사이 주행 시절, 0.24)보다 훨씬 커졌다 -- 실차
# 확인 후 재조정할 것.
AVOID_FULL_OFFSET_RATIO = 1.05

# steer_with_offset(near_offset, far_offset) 호출 시 far 쪽에만 곱하는 배율. 근/원거리에
# 똑같은 오프셋을 주면 Stanley의 heading_error(=(far-near)/half_w)에서 서로 상쇄돼
# 0이 되어버려서, 회피 조향이 K_LATERAL(14)짜리 lateral_error 채널만 쓰게 되고
# 코너처럼 K_HEADING(40)을 쓰는 것보다 훨씬 약한 각도만 나온다(실차 확인: 회피 중
# 최대 조향각이 코너의 1/10 수준). far를 더 크게 줘서 "저 앞 오른쪽 지점을 바라보고
# 튼다"는 헤딩 성분을 만들어 K_HEADING까지 같이 쓰이게 한다.
AVOID_FAR_OFFSET_BOOST = 2.2
# 회피 오프셋 EMA 스무딩 계수. 0.18은 반응이 너무 느려서(실차 확인: 장애물을
# 지나칠 때까지 오프셋이 절반도 못 올라옴) 0.35로 올림 -- 더 적은 프레임 안에
# 목표 오프셋 근처까지 도달해야 실제로 옆으로 비켜나간다.
AVOID_OFFSET_ALPHA = 0.35
AVOID_OFFSET_DONE_RATIO = 0.018  # 오프셋이 이 비율 이하로 줄면 '복귀 완료'로 판정 (0~1)

AVOID_PASS_X_M = 0.10        # 이 거리(전후) 이내를 "바로 옆을 지나는 구간"으로 보고 회피량을 최대로 유지
# 회피 중일 때(아직 크게 안 피한 상태) 속도 상한 (m/s). SPEED_MAX(1.08)보다 뚜렷하게
# 낮게 유지해야 회피 모드에 들어가는 순간 항상 감속이 걸린다 -- SPEED_MAX와 같은
# 값이면 회피 시작부터 옆으로 빠지는 조작 중에 감속이 전혀 안 걸림.
AVOID_CRUISE_SPEED = 0.78
AVOID_EMERGENCY_SPEED = 0.22 # 회피량이 최대일 때 속도 하한 (m/s) -- 정지/후진 없음, 이 아래로는 안 내려감

# ============================================================
# 원시 충돌 감지 (콘으로 확정되지 않은 물체도 전방 근접이면 회피방향을 잡는 보강채널)
# ============================================================
# 정지/후진은 팀 결정으로 하지 않음(control/obstacle_avoid.py 참고) -- 그래서 여기엔
# "정지 거리" 같은 값이 없다. 이 물체가 콘 모양 필터(lidar_params.py)를 통과하든
# 안 하든, 아래 거리/여유 안에 들어오면 obstacle_avoid.py가 즉시 회피방향을 잡는다.
RAW_SHIELD_SOFT_HALF_WIDTH_M = 0.21       # '주의'로 볼 중심선 기준 반폭 (m)
# 회피방향 즉시 결정을 시작할 전방거리 (m). 0.70m는 SPEED_MAX=0.75일 때도 순항
# 속도로 달리면 1초가 채 안 돼 통과해버리는 거리라, AVOID_OFFSET_ALPHA의 EMA가 채
# 오르기도 전에 장애물을 지나쳐 "회피가 늦고 각도가 작다"는 문제의 핵심 원인이었다
# (그 문제의 대부분은 AVOID_OFFSET_ALPHA를 0.35로 올린 것만으로도 해소됨). SPEED_MAX
# 를 0.9로 올린 만큼만 비례해서 1.0m로 늘림 (2.0 m/s까지 고려했던 1.30m는 지금
# SPEED_MAX엔 과함 -- 반응거리를 필요 이상으로 당기면 트랙에 장애물이 촘촘할 때
# 오히려 방해가 될 수 있음).
RAW_SHIELD_BRAKE_X_M = 1.0
