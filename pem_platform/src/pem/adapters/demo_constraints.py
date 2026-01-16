from __future__ import annotations

from typing import Any, Dict


def is_feasible(row: Dict[str, Any]) -> bool:
    """Example feasibility constraint.

    This is a stub for demonstration. Replace with tool/process-specific rules.

    Example rule:
    - temperature_C * pressure_mTorr should be below a threshold (purely illustrative).
    """

    try:
        t = float(row.get("temperature_C"))
        p = float(row.get("pressure_mTorr"))
    except Exception:
        return False

    # illustrative only
    return (t * p) <= 9000.0
