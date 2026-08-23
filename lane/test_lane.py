"""lane 전용 격리 테스트 하니스 -- 장애물/신호등 없이 lane 로직만 확인.

용도: lane 담당자가 속도/조향/코너 진입 동작을 순수하게 관찰할 때. obstacle/avoider와
traffic_light를 아예 부르지 않아서, 출력되는 steer/speed/curve_direction이 lane 로직만의
결과입니다. 실제 주행 검증은 ./run.sh(풀 스택)로 하세요.

실행:
    python3 -u lane/test_lane.py
웹 뷰: http://localhost:5000  (BEV/차선/코너 시각화 -- debug_view.draw_debug_panel 재사용)
종료: Ctrl+C
"""
import time

import versions
import car_api
import config
import debug_view


def main():
    print("lane 격리 테스트 -- 장애물/신호등 미사용, 차량 구동 안 함")
    keeper = versions.lane.LaneKeeper()
    debug_view.serve()
    debug_view.mark_start()

    frame_n = 0
    try:
        while True:
            t0 = time.time()
            img = car_api.camera(config.CAMERA_WIDTH_DRIVE)
            if img is None:
                time.sleep(0.05)
                continue

            # lane 로직만 -- obstacle/traffic 호출 없음
            line_steer, speed, curve_direction, status = keeper.step(img)

            # 순수 lane 출력만 출력 (obstacle bias 섞이지 않음)
            print(f"{debug_view.elapsed_str()}  steer {line_steer:+.1f}  "
                  f"speed {speed:.2f}  curve {curve_direction}  {status}")

            frame_n += 1
            if frame_n % 3 == 0:
                panel = debug_view.draw_debug_panel(img, keeper)
                debug_view.update_web(panel, status)

            proc_time = time.time() - t0
            time.sleep(max(0.0, 1 / config.TARGET_FPS - proc_time))
    except KeyboardInterrupt:
        pass
    finally:
        car_api.stop_vehicle()   # 구동은 안 했어도 정지 명령 한 번 보냄(안전)
        debug_view.stop_view()
        print(f"lane 테스트 종료 -- 총 {debug_view.elapsed_str()}")


if __name__ == "__main__":
    main()
