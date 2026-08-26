# 카메라 파이프라인이 쓰는 모든 값: BEV(조감도) 변환 기하값, 차선 색상 마스크 임계값,
# 신호등 색상/위치 판정 임계값.
#
# 카메라 거치 각도가 바뀌면 ROI_NORM / NEAR_Y_RATIO / FAR_Y_RATIO (필요하면
# FIXED_ROI_NORM도) 를 다시 잡아야 함 -- 이 값을 실제로 소비하는 코드는 sensors/camera.py.
import numpy as np

# ============================================================
# BEV / ROI (원근변환으로 위에서 내려다본 것처럼 바꾸는 영역)
# ============================================================
ROI_NORM = np.float32([
    [0.33, 0.57],   # 좌상단 (정규화 좌표 0~1)
    [0.67, 0.57],   # 우상단
    [0.98, 0.92],   # 우하단
    [0.02, 0.92],   # 좌하단
])
NEAR_Y_RATIO = 0.82   # BEV 변환 후 '가까운 지점' 샘플링 y위치 비율 (0~1, 1에 가까울수록 차량과 가까움)
FAR_Y_RATIO = 0.52    # BEV 변환 후 '먼 지점' 샘플링 y위치 비율 (0~1)

# ============================================================
# 흰색 차선 마스크
# ============================================================
WHITE_S_MAX = 72                   # 흰색으로 인정할 최대 채도 (HSV Saturation, 0~255)
WHITE_V_MIN = 145                  # 흰색으로 인정할 최소 명도 (HSV Value, 0~255)
WHITE_RGB_SPREAD_MAX = 48          # R/G/B 채널값 최대 편차 (0~255) -- 흰색은 채널간 차이가 작아야 함
WHITE_LAB_A_TOL = 20               # Lab 색공간 a채널(초록-빨강) 허용 오차 (0~255 스케일)
WHITE_LAB_B_TOL = 24               # Lab 색공간 b채널(파랑-노랑) 허용 오차 (0~255 스케일)
WHITE_LOCAL_CONTRAST_MIN = 10      # 주변 대비 최소 밝기차 (0~255) -- 아스팔트와 밝기 차이 요구치
WHITE_TOPHAT_KERNEL = 25           # 탑햇 필터(밝은 부분 강조) 커널 크기 (px)
WHITE_ADAPTIVE_V_MARGIN = 38       # 적응형 명도 임계값 여유폭 (0~255)
WHITE_ADAPTIVE_V_MAX = 242         # 적응형 명도 임계값 상한 (0~255)

ASPHALT_S_MAX = 125                # 아스팔트(도로 배경)로 볼 최대 채도 (0~255)
ASPHALT_ADAPTIVE_V_MARGIN = 42     # 아스팔트 적응형 명도 임계값 여유폭 (0~255)
ASPHALT_ADAPTIVE_V_MIN = 135       # 아스팔트 적응형 명도 임계값 하한 (0~255)
ASPHALT_ADAPTIVE_V_MAX = 248       # 아스팔트 적응형 명도 임계값 상한 (0~255)
ROAD_CONTEXT_DILATE = 21           # 도로 영역 팽창(dilate) 연산 커널 크기 (px)

MORPH_KERNEL = 3                   # 마스크 노이즈 제거(open/close) 커널 크기 (px)

# ============================================================
# 흰색 양쪽 경계선 슬라이딩 윈도우 탐색
# ============================================================
NWINDOWS = 9                       # 세로로 나눌 슬라이딩 윈도우 개수 (개)
WINDOW_MARGIN_RATIO = 0.09         # 슬라이딩 윈도우 좌우 반폭 비율 (화면 너비 대비, 0~1)
MINPIX = 18                        # 윈도우 중심 재조정에 필요한 최소 픽셀 수 (개)
MIN_FIT_PIXELS = 70                # 곡선 피팅을 시도할 최소 픽셀 수 (개)

LEFT_SEARCH = (0.03, 0.48)         # 왼쪽 차선 탐색 x범위 (화면 너비 비율, 0~1)
RIGHT_SEARCH = (0.52, 0.97)        # 오른쪽 차선 탐색 x범위 (화면 너비 비율, 0~1)

# 왼쪽 흰선~노란선 사이 코리도 폭 (예전엔 좌우 흰선 사이 전체 차선폭이었으나, 기본
# 주행선을 왼쪽 차로로 좁히면서 폭도 대략 절반으로 줄었다 -- 실차 측정 후 재조정할 것).
LANE_WIDTH_MIN_RATIO = 0.12        # 정상 코리도 폭 최소 비율 (화면 너비 대비)
LANE_WIDTH_MAX_RATIO = 0.55        # 정상 코리도 폭 최대 비율 (화면 너비 대비)
LANE_WIDTH_ALPHA = 0.20            # 코리도 폭 추정값 EMA 스무딩 계수 (0~1, 클수록 최근 값 반영 큼)

# ============================================================
# 노란색 코너 가이드선 마스크
# ============================================================
YELLOW_H_MIN = 5                   # 노란색 최소 색상값 (OpenCV Hue, 0~179)
YELLOW_H_MAX = 35                  # 노란색 최대 색상값 (OpenCV Hue, 0~179)
YELLOW_S_MIN = 110                 # 노란색 최소 채도 (0~255)
YELLOW_V_MIN = 100                 # 노란색 최소 명도 (0~255)
YELLOW_MIN_AREA = 12               # 노란 점으로 인정할 최소 면적 (px^2)
YELLOW_MAX_AREA = 5000             # 노란 점으로 인정할 최대 면적 (px^2)

ASPHALT_V_MAX = 175                # 노란색 후보 주변이 아스팔트여야 하는 최대 명도 (0~255)
ASPHALT_DILATE = 17                # 노란색 후보 주변 아스팔트 판정 팽창 커널 크기 (px)

# 노란 점이 이 개수 미만이면 LaneTracker.detect()가 코리도 오른쪽 경계(yellow_near/
# yellow_far)로 신뢰하지 않는다. 점 1개짜리는 near/far가 항상 같은 값으로 보간되어
# (np.interp가 점 1개면 상수를 반환) 코너 진입 직전처럼 노이즈에 민감한 순간에
# center_near/far를 엉뚱한 방향으로 끌고 갈 수 있다 -- control_params.
# CORNER_PATH_MIN_POINTS와 같은 이유, 다만 이건 STRAIGHT 상태에서도 매 프레임
# 도는 Stanley 조향(center_near/far) 쪽 게이트.
YELLOW_CORRIDOR_MIN_POINTS = 2

# ============================================================
# 노란색 점 -> 경로 연결 (체이닝 기하값. "어디를 목표점으로 볼지"는
# control_params.py의 CORNER_LOOKAHEAD_PX -- 그건 제어 판단이라 여기 안 둠)
# ============================================================
YELLOW_PATH_MAX_LINK_PX = 185.0        # 경로 점들을 연결할 최대 거리 (px)
YELLOW_PATH_FIRST_LINK_PX = 190.0      # 첫 연결(차량-첫점) 최대 거리 (px)
YELLOW_PATH_BACKWARD_ALLOW_PX = 35.0   # 역방향(화면상 뒤로) 연결 허용 거리 (px)

# ============================================================
# 신호등: 카메라 고정 자세 + 고정 탐색 ROI
# ============================================================
DRIVE_CAMERA_PAN_DEG = 0.0         # 주행 중 카메라 pan(좌우) 고정 각도 (도, deg)
DRIVE_CAMERA_TILT_DEG = 0.0        # 주행 중 카메라 tilt(상하) 고정 각도 (도, deg)
CAMERA_SETTLE_SEC = 0.30           # 카메라 자세 명령 후 안정화 대기 시간 (초)

FIXED_ROI_NORM = (0.55, 0.35, 0.98, 0.95)   # 신호등 탐색 고정 영역 (x1,y1,x2,y2, 정규화 좌표 0~1)

NO_LOCK_TIMEOUT_SEC = 5.0          # RED를 한 번도 못 찾을 때 포기하고 강제 출발하기까지 대기 시간 (초)

# ============================================================
# 신호등: RED-LOCK 방식 (TTTTTT_physicar_ros2_red_lock_myapp.py 이식)
# ============================================================
# GREEN은 보지 않는다 -- RED가 RED_CONFIRM_SEC 동안 지속되면 그 순간의 위치+크기를
# 락(lock)하고, 이후엔 그 락된 패치 안에서만 RED가 사라졌는지 본다(다른 곳의 빨간
# 반사광/노이즈에 흔들리지 않기 위함). 락된 패치 안 RED가 RED_RELEASE_SEC 동안
# 사라진 채로 유지되면 출발. decision/traffic_judge.py의 _RedLockGate가 사용.
TRAFFIC_RED_LOW_1 = (0, 100, 100)      # HSV 빨강 구간 1 하한 (Hue 0 부근)
TRAFFIC_RED_HIGH_1 = (10, 255, 255)    # HSV 빨강 구간 1 상한
TRAFFIC_RED_LOW_2 = (170, 100, 100)    # HSV 빨강 구간 2 하한 (Hue 180 부근, 색상환 반대쪽 랩어라운드)
TRAFFIC_RED_HIGH_2 = (179, 255, 255)   # HSV 빨강 구간 2 상한

TRAFFIC_RED_MIN_AREA = 20.0        # 빨강으로 볼 최소 컨투어 면적 (px^2)
TRAFFIC_RED_MIN_RATIO = 0.0010     # 빨강으로 볼 최소 픽셀 비율 (검사 영역 대비, 0~1)
TRAFFIC_MIN_ROI_MEAN_V = 20.0      # 유효 프레임으로 볼 최소 평균 명도 (0~255, 너무 어두우면 무효)
TRAFFIC_MORPH_KERNEL_SIZE = 3      # 마스크 노이즈 제거(open/close) 커널 크기 (px)

TRAFFIC_RED_CONFIRM_SEC = 0.25     # RED를 이 시간 이상 연속 확인해야 락을 건다 (초)
TRAFFIC_RED_RELEASE_SEC = 0.35     # 락된 패치에서 RED가 이 시간 이상 사라져 있어야 출발 (초)

TRAFFIC_LOCK_PADDING_PX = 10           # 락 영역 확장 여백 최소값 (px)
TRAFFIC_LOCK_PADDING_RATIO = 0.60      # 락 영역 확장 여백 비율 (컨투어 bbox 크기 대비)
TRAFFIC_LOCK_RETAIN_AREA_RATIO = 0.18  # 락 후 "여전히 RED"로 볼 최소 면적 비율 (락 당시 면적 대비)
TRAFFIC_LOCK_RETAIN_PIXEL_RATIO = 0.12 # 락 후 "여전히 RED"로 볼 최소 픽셀수 비율 (락 당시 픽셀수 대비)
TRAFFIC_LOCK_MIN_RED_AREA = 8.0        # 락 판정 최소 면적 하한 (px^2, 비율 계산과 무관하게 적용되는 바닥값)
TRAFFIC_LOCK_MIN_RED_PIXELS = 12       # 락 판정 최소 픽셀수 하한 (개, 비율 계산과 무관하게 적용되는 바닥값)
