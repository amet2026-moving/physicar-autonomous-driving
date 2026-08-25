# VehicleMode.RECOVERY용 후진+재탐색 컨트롤러.
#
# TODO: T_T.py main()의 V16 BACKUP -> RECHECK 상태머신을 포팅할 것 (그 안의
# recovery_state / recovery_until / recovery_count / recovery_stall_* 지역변수들이
# 대응됨). 튜닝 상수는 config/control_params.py (RECOVERY_*)에 있음.


class RecoveryController:
    """후진 리커버리 상태를 프레임간에 들고 있는 컨트롤러 (인스턴스 하나를 매 프레임 재사용)."""

    def __init__(self):
        raise NotImplementedError

    def step(self, fusion_result) -> tuple[float, float, str]:
        """이번 프레임의 (조향각(도), 속도(m/s), status)를 반환.
        status는 "ACTIVE"(후진/재확인 진행중) / "DONE"(복귀 완료) 중 하나.
        TODO."""
        raise NotImplementedError
