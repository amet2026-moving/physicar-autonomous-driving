#!/usr/bin/env python3
"""
AME 2026 PhysiCar - LiDAR raw probe

목적:
- 차량을 움직이지 않고 현재 LiDAR의 실제 angle / distance를 확인
- +angle이 실제로 왼쪽인지, -angle이 오른쪽인지 확인
- 정면/좌/우 라바콘이 몇 도 범위와 몇 m 거리로 잡히는지 기록

실행 위치:
    ~/physicar_ws/moving

실행:
    python3 lidar_probe.py

주의:
- main.py / ./run.sh 는 동시에 실행하지 말 것.
- 이 파일은 시작할 때 stop_vehicle()만 1회 보내고, 이후 주행 명령은 보내지 않음.
"""

import math
import time

# 중요:
# car_api.py는 `import config`를 사용하므로 versions.py를 먼저 import해야
# 현재 선택된 config 버전이 sys.modules["config"]에 등록됩니다.
import versions  # noqa: F401
import car_api


PRINT_PERIOD_SEC = 0.70
FRONT_HALF_ANGLE_DEG = 60.0
CENTER_HALF_ANGLE_DEG = 10.0
PRINT_MAX_RANGE_M = 2.50
NEAREST_POINT_COUNT = 12


def polar_to_vehicle(angle_deg, distance_m):
    """
    보기 편한 좌표:
      lateral_left_m: 왼쪽이 +
      forward_m: 전방이 +
    """
    rad = math.radians(float(angle_deg))
    lateral_left_m = float(distance_m) * math.sin(rad)
    forward_m = float(distance_m) * math.cos(rad)
    return lateral_left_m, forward_m


def sector_min(points, lo_deg, hi_deg):
    selected = [(a, d) for a, d in points if lo_deg <= a <= hi_deg]
    if not selected:
        return None
    return min(selected, key=lambda p: p[1])


def fmt_point(point):
    if point is None:
        return "----"
    a, d = point
    left, forward = polar_to_vehicle(a, d)
    return (
        f"a={a:+6.1f}°  d={d:5.2f}m  "
        f"left={left:+5.2f}m  forward={forward:5.2f}m"
    )


def main():
    # 이전 실행 명령이 남아 있지 않도록 차량 정지.
    car_api.stop_vehicle()

    print("=" * 78)
    print("LiDAR RAW PROBE")
    print("차량은 움직이지 않습니다.")
    print("현재 car_api 규약: +angle = 왼쪽 / -angle = 오른쪽")
    print("Ctrl+C 로 종료")
    print("=" * 78)

    try:
        while True:
            points = car_api.lidar()

            if not points:
                print("\n[NO DATA] lidar() returned empty list")
                time.sleep(PRINT_PERIOD_SEC)
                continue

            # 전방 ±60° 중, 너무 먼 배경은 출력에서 제외.
            front = [
                (float(a), float(d))
                for a, d in points
                if abs(float(a)) <= FRONT_HALF_ANGLE_DEG
                and float(d) <= PRINT_MAX_RANGE_M
            ]

            center = sector_min(
                front,
                -CENTER_HALF_ANGLE_DEG,
                CENTER_HALF_ANGLE_DEG,
            )
            left = sector_min(front, +10.0, +60.0)
            right = sector_min(front, -60.0, -10.0)
            nearest = min(front, key=lambda p: p[1]) if front else None

            print("\n" + "-" * 78)
            print(f"valid points={len(points)} / front<=2.5m={len(front)}")
            print(f"CENTER ±10° : {fmt_point(center)}")
            print(f"LEFT +10~60°: {fmt_point(left)}")
            print(f"RIGHT -60~-10°: {fmt_point(right)}")
            print(f"NEAREST ±60°: {fmt_point(nearest)}")

            # 가까운 점부터 보여주면 콘이 실제로 어느 각도 묶음으로 잡히는지 바로 볼 수 있음.
            print(f"\nclosest {NEAREST_POINT_COUNT} front points:")
            for a, d in sorted(front, key=lambda p: p[1])[:NEAREST_POINT_COUNT]:
                lateral_left, forward = polar_to_vehicle(a, d)
                side = "L" if a > 1e-6 else "R" if a < -1e-6 else "C"
                print(
                    f"  {side}  angle={a:+6.1f}°"
                    f"  dist={d:5.2f}m"
                    f"  left={lateral_left:+5.2f}m"
                    f"  forward={forward:5.2f}m"
                )

            time.sleep(PRINT_PERIOD_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        car_api.stop_vehicle()
        print("\nprobe stopped")


if __name__ == "__main__":
    main()
