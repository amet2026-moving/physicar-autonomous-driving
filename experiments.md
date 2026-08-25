# 실험 기록

실차 테스트 기록. 최신순으로 위에 추가. 로그 파일 자체는
`logs/run_{YYYYMMDD}_{HHMMSS}.log` (utils/logger.py 참고)에 저장되니, 여기에는
run_id만 남겨서 결과와 전체 로그를 서로 연결할 수 있게 한다.

옛 방식(config 파일에 주석블록을 계속 쌓아가는 것)을 대신함 -- 마크다운으로 따로
빼두면 실패한 실험이 죽은 코드로 계속 쌓이지 않는다.

## 템플릿

```
## YYYY-MM-DD HH:MM -- run_YYYYMMDD_HHMMSS
- 변경사항: (지난 기록 이후 바뀐 config/코드)
- 결과: 랩타임 N초 / 충돌 N회 / 트랙이탈 N회
- 다음: (다음에 시도하거나 고칠 것)
```

## 구현 진행 체크리스트 (다 채우면 이 섹션은 지울 것)

- [x] `sensors/camera.py` -- BEV/ROI + 흰선 + 노란선 + 신호등 인식 (draw_debug 포함, self-test: `python -m sensors.camera`)
- [x] `sensors/lidar.py` -- 스캔 수신(rclpy) + 클러스터링 + 콘 형상 분류 (self-test: `python -m sensors.lidar`)
- [ ] `sensors/fusion.py` -- 차선기준 장애물 좌/우 판정
- [ ] `decision/*_judge.py` -- 신호등/차선/장애물 상태 판정
- [ ] `control/lane_follow.py` -- Stanley 조향 + 속도계획
- [ ] `control/obstacle_avoid.py` -- 회피 컨트롤러
- [ ] `control/recovery.py` -- 후진/재확인 컨트롤러
- [ ] `control/modes.py` -- VehicleMode 전이 규칙
- [ ] `myapp/debug_view.py` build_panel() -- 통합 시각화
- [ ] `utils/rosbag_record.py` / `rosbag_replay.py`
