# PhysiCar 자율주행 -- end2end 진입점.
#
# 이 파일은 파이프라인을 "조립"만 함. 인식/판단/제어 로직을 여기에 직접 쓰면 안 됨 --
# 그건 sensors/, decision/, control/ 쪽 책임. 여기서 if/elif를 쌓고 있다면, 그 로직은
# control/modes.py나 *_judge.py로 옮겨야 한다는 신호.
#
# 프레임당 파이프라인:
#   sensors.camera / sensors.lidar  ->  sensors.fusion
#     -> decision.*_judge            ->  control.modes.decide_mode
#     -> control.lane_follow / obstacle_avoid (현재 모드에 맞는 것 -- recovery 없음)
#     -> hardware.car_api.drive()
#
# 지금은 아래 sensors/decision/control 호출들이 전부 NotImplementedError를 던짐 --
# 정상임. experiments.md의 계획대로 sensors/ -> decision/ -> control/ 순서로 채워나가면
# 하나씩 사라진다.
import time

from config import base as cfg
from hardware import car_api
from myapp import debug_view
from sensors import camera, lidar, fusion
from decision import traffic_judge, lane_judge, obstacle_judge
from control import modes, lane_follow
from control.obstacle_avoid import ObstacleAvoidController
from utils import logger
from utils.states import VehicleMode, TrafficLightState


def main():
    logger.setup_run_logging()          # 터미널 출력을 logs/run_*.log 에도 동시 기록 시작
    logger.install_shutdown_handlers()  # SIGTERM을 Ctrl+C처럼 처리해서 finally 정리가 항상 실행되게 함

    debug_view.serve()   # 웹 디버그 뷰 서버 시작 (http://localhost:5000)

    if cfg.DRIVE_ENABLED:
        car_api.stop_vehicle()   # 이전 실행에서 남은 명령이 있으면 정리

    print("=" * 72)
    print("PhysiCar autonomous driving")
    print(f"DRIVE_ENABLED = {cfg.DRIVE_ENABLED}")
    if not cfg.DRIVE_ENABLED:
        print("DRIVE_ENABLED=False -- debug view only, not driving.")
    print("Ctrl+C to stop.")
    print("=" * 72)

    avoid_ctrl = ObstacleAvoidController()   # 장애물 회피 상태를 프레임간 유지하는 컨트롤러 인스턴스

    mode = VehicleMode.LANE_FOLLOW   # 최초 진입 모드
    current_speed = 0.0              # 현재 속도 (m/s) -- 다음 프레임 가감속 램프 계산에 쓰임
    start_time = time.time()         # 경과시간 로그 출력을 위한 시작 시각
    frame_n = 0                      # 처리한 프레임 수 카운터

    try:
        if cfg.ENABLE_TRAFFIC_LIGHT_WAIT:
            traffic_judge.wait_for_departure()   # RED가 나타났다 사라질 때까지 블로킹 대기 (루프 시작 전 1회)

        while True:
            t0 = time.time()   # 이번 프레임 처리 시작 시각 (처리시간/Hz 계산용)

            frame = car_api.camera()      # 원본 카메라 프레임 (BGR ndarray)
            if frame is None:
                # 카메라 요청 실패(타임아웃/네트워크 순간끊김) -- wait_for_departure()와
                # 같은 방식으로 정지 유지하고 다음 프레임을 재시도한다. 여기서 그냥
                # 진행하면 frame.shape 접근에서 죽는다.
                car_api.stop_vehicle()
                time.sleep(0.10)
                continue

            scan = lidar.capture_scan()   # 원시 라이다 스캔

            bev_frame, _ = camera.warp_to_bev(frame)          # 원근변환으로 위에서 내려다본 이미지로 변환
            lane_obs = camera.detect_lane_lines(bev_frame)    # 흰색 차선 인식 결과

            clusters = lidar.build_clusters(scan)                       # 라이다 포인트를 물체 단위로 클러스터링
            fusion_result = fusion.classify_obstacle_side(lane_obs, clusters)  # 차선기준 좌/우 장애물 판정

            # 이번 프레임의 관측 결과를 각각의 상태(Enum)로 판정
            # judge_lane_state()는 코너 히스테리시스에 쓸 "직전 프레임이 코너였는지"를
            # 자기 내부에 기억해두므로(decision/lane_judge.py 참고) 여기서 넘길 필요 없음.
            lane_state = lane_judge.judge_lane_state(lane_obs)
            obstacle_state = obstacle_judge.judge_obstacle_state(fusion_result)

            avoid_status = "DONE"   # 회피 모드가 아닐 때의 기본값 (다음 모드 결정에 쓰임)
            if mode == VehicleMode.OBSTACLE_AVOID:
                steer, current_speed, avoid_status = avoid_ctrl.step(
                    fusion_result, clusters, lane_obs,
                )
                # lane_follow의 조향 EMA를 계속 동기화해둔다 -- 안 하면 회피 끝나고
                # LANE_FOLLOW로 복귀할 때 회피 시작 전의 오래된 값에서부터 다시
                # 스무딩을 시작하게 되어 복귀 순간 조향이 잠깐 어긋난다.
                lane_follow.sync_steer(steer)
            else:
                steer, current_speed = lane_follow.compute(lane_obs, lane_state, current_speed)

            mode = modes.decide_mode(mode, lane_state, obstacle_state, avoid_status)   # 다음 프레임에 쓸 모드 결정

            # 웹 디버그 뷰 갱신. 신호등은 wait_for_departure()에서 루프 시작 전 1회만
            # 판정하므로(주행 중엔 다시 안 봄), camera.draw_debug()에는 빈 값을 넘겨서
            # D 패널이 게이트 통과 시점 상태로 고정 표시되게 한다 -- 매 프레임 신호등
            # ROI를 다시 스캔하는 불필요한 연산을 피하기 위함.
            camera_panel = camera.draw_debug(frame, bev_frame, lane_obs, camera.TrafficLightObservation())
            lidar_panel = lidar.draw_debug(clusters)
            fusion_panel = fusion.draw_debug(bev_frame, lane_obs, clusters, fusion_result)
            debug_view.update_web(
                debug_view.build_panel(camera_panel, lidar_panel, fusion_panel),
                f"MODE={mode.value} lane={lane_state.value} obstacle={obstacle_state.value} "
                f"steer={steer:+.1f} speed={current_speed:.2f}",
            )

            if cfg.DRIVE_ENABLED:
                car_api.drive(current_speed, steer)   # 실제 조향/속도 명령 전송 (speed: m/s, steer: 도)

            frame_n += 1
            proc_time = time.time() - t0             # 이번 프레임 처리에 걸린 시간 (초)
            hz = 1.0 / max(proc_time, 1e-6)           # 처리 주파수 (Hz)

            if frame_n % 10 == 0:   # 매 프레임 출력하면 너무 많으므로 10프레임마다 한 줄만 출력
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed // 60):02d}:{elapsed % 60:04.1f}"   # 분:초.소수 형식
                # 신호등은 루프 시작 전 wait_for_departure()에서 딱 한 번만 판정됨 -- 여기서
                # GREEN으로 찍는 건 "매 프레임 다시 확인 중"이 아니라 "게이트를 이미 통과했음"을 뜻함
                logger.status_line(
                    elapsed_str, mode, TrafficLightState.GREEN, lane_state, obstacle_state,
                    steer, current_speed, hz,
                )

            time.sleep(max(0.0, 1.0 / cfg.TARGET_FPS - proc_time))   # 목표 FPS에 맞춰 남은 시간만큼 대기

    except KeyboardInterrupt:
        pass   # Ctrl+C -- 아래 finally에서 정상 종료 처리
    finally:
        car_api.stop_vehicle()   # 루프가 어떻게 끝나든 마지막으로 반드시 정지 시도
        debug_view.stop_view()
        print("stopped")


if __name__ == "__main__":
    main()
