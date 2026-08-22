"""PhysiCar 자율주행 메인 루프 -- 2026 AMET 해커톤.

이 파일 하나가 전체 서브시스템(하드웨어 API / 차선+코너 / 장애물회피 / 신호등 / 디버그뷰)을
조립합니다. run.sh가 이 파일을 실행하면 end2end로 주행합니다.

각 서브시스템은 독립된 파일로 분리돼 있습니다 -- 예를 들어 장애물 회피 알고리즘만 바꾸고
싶다면 obstacle_avoidance.py만 건드리면 되고, 차선 인식 알고리즘만 바꾸고 싶다면 lane_tracing.py만 건드리면
됩니다. 단, 코너모드 상태(keeper.corner_active)가 장애물회피 쪽 조향 억제에도 쓰이는 것처럼
서브시스템 사이에 넘어가는 값들이 있으니, 각 모듈 상단 docstring을 참고하세요.

안전 관련 설계는 config.py DRIVE_ENABLED 설명, car_api.py 파일 상단 설명을 먼저 읽어보세요.
"""
import time

import car_api
import config
import debug_view
import lane_keeper
import obstacle_avoidance
import traffic_light


def main():
    print("PhysiCar 자율주행: Stanley 차선추종 + 코너 폴백 + LiDAR 장애물회피 + 신호등")
    keeper = lane_keeper.LaneKeeper()
    avoider = obstacle_avoidance.ObstacleAvoider()
    frame_n = 0

    debug_view.serve()

    if config.DRIVE_ENABLED:
        car_api.stop_vehicle()   # 이전 실행에서 남은 명령 정리

    print("=" * 72)
    print(f"DRIVE_ENABLED = {config.DRIVE_ENABLED}")
    if not config.DRIVE_ENABLED:
        print("DRIVE_ENABLED=False -- 디버그 뷰만 확인하고 실제로는 주행하지 않습니다.")
        print("웹 뷰에서 BEV/차선 인식이 맞는 걸 확인한 뒤에만 config.py에서 True로 바꾸세요.")
    print("Ctrl+C to stop.")
    print("=" * 72)

    try:
        if config.ENABLE_TRAFFIC_LIGHT_WAIT:
            traffic_light.wait_for_green_and_return_to_driving_pose()

        while True:
            t0 = time.time()

            cam_future = car_api.pool.submit(car_api.camera, config.CAMERA_WIDTH_DRIVE)
            lidar_future = car_api.pool.submit(car_api.lidar)
            img = cam_future.result()
            points = lidar_future.result()

            line_steer, speed, curve_direction, status = keeper.step(img)
            result = avoider.step(points, curve_direction, keeper.corner_active, t0)

            if result["stopped_boxed"]:
                steer, speed = 0.0, 0.0
                status += "  STOPPED (boxed in, backup attempts exhausted)"
            elif result["backing"]:
                steer, speed = result["backup_steer"], result["backup_speed"]
                status = (f"BACKUP (boxed in, front {result['min_dist']:.2f}m, "
                          f"attempt {result['backup_attempts']}/{config.MAX_BACKUP_ATTEMPTS})")
            else:
                steer = max(-config.STEER_MAX, min(config.STEER_MAX, line_steer + result["bias"]))
                min_dist = result["min_dist"]
                if min_dist < config.OBSTACLE_SLOWDOWN_RANGE:
                    obstacle_factor = max(config.OBSTACLE_MIN_SPEED / config.SPEED_MAX,
                                           min_dist / config.OBSTACLE_SLOWDOWN_RANGE)
                    speed = min(speed, config.SPEED_MAX * obstacle_factor)
                if min_dist < config.OBSTACLE_MAX_RANGE:
                    status += f"  obstacle {min_dist:.2f}m bias {result['bias']:+.1f}"

            if result["lidar_stale_for"] > config.LIDAR_STALE_GRACE_S:
                speed = min(speed, config.LIDAR_STALE_SPEED_CAP)
                status += f"  LIDAR STALE {result['lidar_stale_for']:.1f}s"

            if config.DRIVE_ENABLED:
                car_api.drive(speed, steer)

            frame_n += 1
            if img is not None and frame_n % 3 == 0:
                panel = debug_view.draw_debug_panel(img, keeper)
                debug_view.update_web(panel, status)

            proc_time = time.time() - t0
            if frame_n % 10 == 0:
                print(f"steer {steer:+.1f}  speed {speed:.2f}  {status}  "
                      f"({1 / max(proc_time, 1e-6):.1f}Hz)")

            time.sleep(max(0.0, 1 / config.TARGET_FPS - proc_time))
    except KeyboardInterrupt:
        pass
    finally:
        # 루프 중 어떤 예외로 끝나든(KeyboardInterrupt 포함) 마지막으로 정지를 시도합니다.
        # car_api.stop_vehicle()은 자체적으로 모든 예외를 삼키므로 여기서 또 죽지 않습니다.
        car_api.stop_vehicle()
        debug_view.stop_view()
        print("stopped")


if __name__ == "__main__":
    main()
