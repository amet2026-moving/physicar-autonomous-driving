"""PhysiCar 자율주행 -- 2026 AMET 해커톤. 전 서브시스템 튜닝 상수 모음.

이 프로젝트는 팀원별로 병렬 개발되던 여러 버전(재모 modi_test_v5/v6, Sangheon test_v7/v8,
성찬 auto3/auto4)을 하나의 협업 구조로 합친 것입니다. 값의 출처:
  - 차선 인식 / Stanley 조향 / 코너 폴백 / 속도계획 / 하드웨어 API / 장애물회피:
    성찬님 auto4.py 기준 (실제 로봇에서 랩타임까지 검증된 가장 최신 튜닝값이라 채택 --
    auto3->auto4에서 스티어링 데드밴드를 롤백한 버전).
  - 신호등(팬/틸트 탐색 + 검정영역 내부 초록만 인식): Sangheon test_v8.py 기준
    (본선 신호등 위치가 랜덤일 수 있어 좌우 대응이 되는 v8 방식을 채택).
  - REAR_HALF_ANGLE / BACKUP_REAR_RANGE / MAX_BACKUP_ATTEMPTS / CORNER_ENTER_DEBOUNCE_FRAMES:
    코드 리뷰에서 발견된 버그(후방 미체크 후진, 코너진입 디바운스 우회)를 고치며 새로 추가.

여러 파일이 같이 참조하는 상수는 "SHARED" 라고 표시해뒀습니다 -- 한쪽 목적으로 값을 바꾸면
다른 서브시스템 동작도 같이 바뀌니, 이 상수들은 바꾸기 전에 팀에 공유하세요.
"""

# ============================================================
# 0. 실행 모드
# ============================================================

# 최초 실행은 반드시 False로 두고 디버그 뷰(웹 패널)에서 BEV/차선 인식이 맞는지 확인한 뒤에만
# True로 바꾸세요. (여러 팀원 버전에서 이 순서를 지키지 않고 True로 커밋된 채 실행되는 사고가
# 반복 확인되어, 여기서는 문서화된 의도대로 기본값을 False로 되돌려놨습니다.)
DRIVE_ENABLED = False

TARGET_FPS = 15.0
PRINT_INTERVAL = 1.0

# 신호등 대기 단계를 건너뛰고 싶을 때(예: 신호등 없는 트랙에서 차선/장애물만 테스트) False로.
ENABLE_TRAFFIC_LIGHT_WAIT = True

# ============================================================
# 1. 하드웨어 / 통신
# ============================================================
BASE_URL = "http://localhost"

STEER_MAX = 20.0          # 하드웨어 한계 (agent.md, physicar-ros2.md)   [SHARED: lane/control/obstacle]
MAX_STEERING_DEG = STEER_MAX   # alias -- lane_tracing.py 쪽 코드가 이 이름으로 참조

CAMERA_WIDTH_DRIVE = 320  # 네트워크/디코딩/CV 연산 시간을 줄이기 위한 축소 해상도

# ============================================================
# 2. 카메라 ROI / BEV
# ============================================================
import numpy as np  # noqa: E402  (ROI_NORM 정의에 필요)

ROI_NORM = np.float32([
    [0.33, 0.57],   # top-left
    [0.67, 0.57],   # top-right
    [0.98, 0.92],   # bottom-right
    [0.02, 0.92],   # bottom-left
])
NEAR_Y_RATIO = 0.82
FAR_Y_RATIO = 0.52

# ============================================================
# 3. 흰색 마스크
# ============================================================
WHITE_S_MAX = 65
WHITE_V_MIN = 150
MORPH_KERNEL = 3

# ============================================================
# 4. 흰색 양쪽 경계선 슬라이딩 윈도우 추적
# ============================================================
NWINDOWS = 9
WINDOW_MARGIN_RATIO = 0.09
MINPIX = 18
MIN_FIT_PIXELS = 70
LEFT_SEARCH = (0.03, 0.48)
RIGHT_SEARCH = (0.52, 0.97)
LANE_WIDTH_MIN_RATIO = 0.30
LANE_WIDTH_MAX_RATIO = 0.95
LANE_WIDTH_ALPHA = 0.20

PREVIEW_MIN_Y_RATIO = 0.44   # 속도계획용 미리보기 샘플링 하한

# ============================================================
# 5. 조향 (Stanley Controller)
# ============================================================
K_LATERAL = 16.0    # Stanley 횡방향 게인
K_HEADING = 40.0    # Stanley 헤딩 게인
STEER_ALPHA = 0.65  # 최종 조향 출력 EMA -- 급코너 반응 속도와 부드러움의 트레이드오프

STANLEY_SOFT_FACTOR = 0.45        # 저속에서 분모가 0에 가까워져 과도하게 튀는 것 방지
STANLEY_CROSS_TRACK_SCALE = 2.2   # 횡오차(atan) 항의 최종 가중치
STANLEY_HEADING_SCALE = 8.0       # 헤딩오차 항의 최종 가중치

# 데드밴드는 의도적으로 0 -- 작은 지속 편향(도로 크라운 등)이 데드밴드 안에서 누적됐다가
# 문턱을 넘는 순간 과보정하는 리밋사이클(limit cycle)이 생기는 게 확인되어 롤백된 값입니다.
# 노이즈 억제는 아래 CENTER_SMOOTH_ALPHA(원인 자체를 스무딩)가 담당합니다.
STEER_DEADBAND_DEG = 0.0

# 차선 중심점(center_near/far) 자체를 프레임간 EMA로 스무딩 -- 직선에서 미세 픽셀 노이즈가
# heading_error 항(가중치가 lateral_error보다 훨씬 큼)을 타고 조향 진동으로 나타나는 걸 방지.
CENTER_SMOOTH_ALPHA = 0.5

# ============================================================
# 6. 코너 폴백 (노란 점선 중앙선 추적)
# ============================================================
# 흰선이 잡히는 동안(직진이든 커브든)은 무조건 흰선 Stanley가 처리하고, 노란선은 흰선을
# 놓쳤거나(weak_white) 조향/헤딩이 이미 커브 수준일 때만 폴백으로 쓰입니다.
YELLOW_H_MIN, YELLOW_H_MAX = 5, 35
YELLOW_S_MIN, YELLOW_V_MIN = 110, 100
YELLOW_MIN_AREA, YELLOW_MAX_AREA = 12, 5000
ASPHALT_V_MAX = 175      # 노란색 후보가 이보다 어두운(=아스팔트) 배경 위에 있어야만 인정
ASPHALT_DILATE = 17

CORNER_ENTER_HEADING = 0.13
CORNER_ENTER_STEER = 9.0
CORNER_ENTER_WEAK_WHITE_FRAMES = 3   # 흰선 LEFT_ONLY/RIGHT_ONLY/LOST가 이 프레임 수만큼
                                      # 연속돼야만 코너진입 조건("weak_white")으로 인정
                                      # -- 콘이 흰선을 1프레임만 가린 것으로 오진입 방지

# [FIX] 원본(auto4)에서는 normal_sharp(heading_error 큼)/normal_big_steer(steering 큼) 조건이
# 디바운스 없이 단일 프레임만으로 코너모드를 켤 수 있었습니다(weak_white만 디바운스 적용).
# 콘이 흰선을 살짝 왜곡시켜 폴리핏이 한 프레임 튀면 이 경로로 여전히 오진입할 수 있어서,
# 이 두 조건에도 동일한 프레임 수 디바운스를 추가로 적용합니다 (lane_keeper.py 참고).
CORNER_ENTER_DEBOUNCE_FRAMES = 3

CORNER_EXIT_HEADING = 0.085
CORNER_EXIT_STEER = 6.0
CORNER_EXIT_BOTH_FRAMES = 4

CORNER_LOOKAHEAD_PX = 125.0
CORNER_MAX_LINK_PX = 185.0
CORNER_FIRST_LINK_PX = 190.0
CORNER_BACKWARD_ALLOW_PX = 35.0
CORNER_STEER_GAIN = 0.70   # 목표각도 -> 조향 변환 게인. 0.70이면 약 28.6도부터 풀 조향(STEER_MAX)
CORNER_HOLD_FRAMES = 7     # 노란 경로가 몇 프레임 사라져도 마지막 코너 조향을 유지(즉시 정지 안 함)

# 코너 진입 직후(조향 작음)엔 SPEED_CORNER 근처, 급할수록(조향 STEER_MAX 근접) 이 값까지
# 조향각 비례로 선형 감속.
SPEED_CORNER_MIN = 0.32

# ============================================================
# 7. 속도 계획
# ============================================================
SPEED_MAX = 0.85            # 실제 로봇 실주행으로 검증된 값 (60초대 랩타임 보고)   [SHARED: obstacle]
SPEED_MIN = 0.44            # 곡선(NORMAL 모드) 최저속도
SPEED_ONE_LINE = 0.40       # 흰선 한쪽만 잡힐 때
SPEED_CORNER = 0.68         # 코너 진입 직후(완만~중간 코너) 속도 상한
SPEED_CORNER_HOLD = 0.55    # 코너 중 노란 경로가 잠깐 사라졌을 때(hold) 속도
SPEED_RAMP_UP = 0.30        # 프레임당 최대 가속
SPEED_RAMP_DOWN = 0.08      # 프레임당 최대 감속 (감속은 보수적으로 유지)

CURVE_DEADBAND = 0.30
STRAIGHT_CURVE_LIMIT = 0.24   # 이 값 이하 곡률은 "직선급"으로 보고 SPEED_MAX 유지
TARGET_SPEED_ALPHA = 0.30     # 목표속도 자체를 스무딩 (급격한 목표속도 변화 완화)

# ============================================================
# 8. 완전히 놓쳤을 때 (LOST 복구 -- 정지 대신 느린 크리핑 재탐색)
# ============================================================
# 대회 중 사람 개입이 불가능해서, 흰선/노란선을 모두 완전히 놓쳤을 때 그냥 정지하는 대신
# 마지막 조향 방향으로 느리게 크리핑하며 재탐색합니다 (Stateless 안전장치).
SEARCH_STEER = 10.0   # OBSTACLE_BIAS_MAX(18.0)보다 작아야 위급 시 회피가 반드시 이김
SEARCH_SPEED = 0.28   # 눈이 없는 상태로 크리핑하는 거라 SPEED_MIN보다는 낮게 유지

# ============================================================
# 9. LiDAR 장애물 회피 + 후진 탈출
# ============================================================
LIDAR_STEP_DEG = 2

OBSTACLE_MAX_RANGE = 1.90        # 이 거리부터 조향 보정 시작 (일찍부터 반응해서 옆 여유 확보)
OBSTACLE_FRONT_HALF_ANGLE = 60
OBSTACLE_STEER_GAIN = 55.0
OBSTACLE_BIAS_MAX = 18.0         # STEER_MAX(20도)에 가깝게 -- 직선에서는 line_steer가 거의
                                  # 0이라 회피가 쓸 수 있는 각도 여유가 원래 큼
OBSTACLE_MIN_SPEED = 0.50        # 장애물에 가장 가까이 붙었을 때의 최저속도
OBSTACLE_SLOWDOWN_RANGE = 1.00   # 이 거리부터 감속 시작 (조향 보정 시작 거리와는 분리)
OBSTACLE_BIAS_ALPHA = 0.5        # 회피 보정치 EMA -- 라이다 프레임간 노이즈로 조향이 안 떨리게

LEFT_ROOM_ANGLES = (15, 90)
RIGHT_ROOM_ANGLES = (-90, -15)
ROOM_HYSTERESIS_M = 0.15
CURVE_OVERRIDE_MARGIN = 0.30   # 차선이 아는 방향이 반대쪽보다 이만큼 더 막혀있어야만 override
LINE_CURVE_THRESHOLD = 5.0     # 이 이상 조향 중이면 회피 방향 힌트로 신뢰 (보조 신호)
LANE_BIAS_DEADBAND = 0.15      # |lateral_error|가 이보다 크면 그 반대쪽을 회피 힌트로 최우선
LANE_BIAS_ALPHA = 0.4          # lateral_error 스무딩 -- 방향 힌트가 프레임마다 안 뒤집히게

# 코너 중(corner_active)엔 원칙적으로 회피를 조향에 반영하지 않고 감속만 함. 단 장애물이
# CORNER_OBSTACLE_EMERGENCY_DIST보다 바짝 붙으면 예외적으로 조향에도 반영.
CORNER_OBSTACLE_STEER = False
CORNER_OBSTACLE_EMERGENCY_DIST = 0.55

BACKUP_FRONT_RANGE = 0.25   # 정면이 이보다 가까우면 "코앞에 막힘" 후보
BACKUP_SIDE_RANGE = 0.35    # 좌우 여유도 이보다 좁으면(=피할 각도 안 나옴) 후진 트리거
BACKUP_SPEED = -0.20
BACKUP_DURATION_S = 1.0

# [FIX] 원본(auto4)에는 후방 라이다 체크가 전혀 없어서, boxed_in 후진이 뒤쪽 장애물로 바로
# 후진할 수 있고, 전방이 안 뚫리면 무한 재트리거될 수 있었습니다. 후방 체크 + 후진 시도 상한을
# 추가합니다 (obstacle_avoidance.py 참고). 각도는 lidar()가 (-180,180]로 정규화하므로 뒤쪽은 +180/-180
# 양끝에 걸쳐 있음 -- abs(angle) >= 180 - REAR_HALF_ANGLE 로 양쪽 끝을 함께 잡습니다.
REAR_HALF_ANGLE = 30
BACKUP_REAR_RANGE = 0.25
MAX_BACKUP_ATTEMPTS = 3   # 전방이 안 뚫린 채로 연속 후진 허용 횟수 -- 초과하면 후진 대신 정지

# 짧은 라이다 통신 끊김(수 프레임)은 마지막 유효 스캔을 재사용해 매끄럽게 넘기되, 그보다
# 오래 끊기면 위치가 이미 바뀌었을 스캔을 계속 믿지 않고 방향판단은 포기, 속도만 제한.
LIDAR_STALE_GRACE_S = 0.5
LIDAR_STALE_SPEED_CAP = 0.35

# ============================================================
# 10. 하이브리드: 학습된 CNN으로 차선 추종 대체 (없으면 자동 폴백)
# ============================================================
USE_LEARNED_LANE_MODEL = False
LANE_MODEL_PATH = "lane_model.onnx"
LANE_MODEL_WIDTH, LANE_MODEL_HEIGHT = 160, 120

# ============================================================
# 11. 신호등 (팬/틸트 탐색 + 검정 영역 내부 초록만 인식 -- test_v8 기준)
# ============================================================
# 본선 신호등 위치가 랜덤일 수 있어 고정 ROI 대신 화면 대부분을 탐색합니다.
GREEN_CONFIRM_FRAMES = 3
TRAFFIC_ROI_NORM = (0.02, 0.02, 0.98, 0.88)   # 하단 12%는 도로/차선 오탐 방지로 제외

# 주행 자세(0/0)에서 차선 ROI가 검증되었으므로, 주행 시작 전 반드시 이 자세로 복귀합니다.
DRIVE_CAMERA_PAN_DEG = 0.0
DRIVE_CAMERA_TILT_DEG = 0.0

# 카메라 pan/tilt 하드웨어 범위(±30도) 안에서만 명령. 정면부터 찾고, 없으면 좌/우/위쪽 순.
TRAFFIC_SEARCH_POSES = [
    (0.0, 0.0),
    (-20.0, 0.0),
    (20.0, 0.0),
    (-30.0, 0.0),
    (30.0, 0.0),
    (0.0, 15.0),
    (-20.0, 15.0),
    (20.0, 15.0),
    (-30.0, 15.0),
    (30.0, 15.0),
]

# 서보 정착시간 -- /joint_states 기반 실제 각도 확인은 후속 실차 검증에서 필요 시 추가.
CAMERA_SETTLE_SEC = 0.30
TRAFFIC_POSE_DWELL_SEC = 0.60
TRAFFIC_LOCK_FRAMES = 2
TRAFFIC_LOCK_LOST_SEC = 1.20

# OpenCV HSV: H=[0,179], S/V=[0,255]
TRAFFIC_RED_H1 = (0, 12)
TRAFFIC_RED_H2 = (168, 179)
TRAFFIC_RED_S_MIN = 145
TRAFFIC_RED_V_MIN = 155

TRAFFIC_GREEN_H = (38, 92)
TRAFFIC_GREEN_S_MIN = 120
TRAFFIC_GREEN_V_MIN = 150

# BGR 채널 우세 판정 (HSV만으로는 잔디 초록과 혼동될 수 있어 추가 필터)
TRAFFIC_RED_CHANNEL_MIN = 135
TRAFFIC_RED_DOMINANCE = 45
TRAFFIC_GREEN_CHANNEL_MIN = 135
TRAFFIC_GREEN_DOMINANCE = 35

# Blob/문맥 필터
TRAFFIC_MIN_BLOB_AREA_RATIO = 0.00005
TRAFFIC_MAX_BLOB_AREA_RATIO = 0.080
TRAFFIC_MIN_BBOX_FILL = 0.38
TRAFFIC_MIN_CIRCULARITY = 0.28
TRAFFIC_EDGE_MARGIN_PX = 3

# 신호등 램프 주변은 검정 하우징이라는 특징 -- 색상 후보가 어두운 영역에 둘러싸여 있어야만 인정
TRAFFIC_DARK_V_MAX = 95
TRAFFIC_DARK_EXPAND = 2.1
TRAFFIC_MIN_DARK_SURROUND_RATIO = 0.25
TRAFFIC_MIN_RING_PIXELS = 30

TRAFFIC_SCORE_MIN = 0.50
TRAFFIC_GREEN_OVER_RED_MARGIN = 0.08
