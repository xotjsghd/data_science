"""공통 데이터 로딩: data/hourly.parquet → 설비별/공장 전체 시간 단위 시리즈 (kWh)"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# 원본 CSV·공모전 당시 결과 파일이 있는 폴더. 환경변수 RAW_DATA_DIR로 지정하고, 없으면 이 프로젝트의 상위 폴더
RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", ROOT.parent))
HOURLY_PATH = ROOT / "data" / "hourly.parquet"

# 학습·예측 기간에 걸친 휴일 (5/1 근로자의 날, 5/5 어린이날·부처님오신날, 5/6 대체공휴일 포함)
HOLIDAYS = pd.to_datetime([
    "2024-12-25", "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-03-01", "2025-03-03", "2025-05-01", "2025-05-05", "2025-05-06",
])

MIN_COVERAGE = 0.5  # 한 시간에 5초 샘플이 절반도 없으면 그 시간 평균은 신뢰하지 않음


def load_module_hourly() -> pd.DataFrame:
    """
    설비×시간 wide 테이블 (index=dt_hour, columns=module, 값=kWh).
    가이드 정의(activePower = kW)대로 activePower 시간평균(kW) × 1h = 그 시간의 전력량(kWh).
    공모전 제출본·주최측 정답과 같은 스케일이며 accumActiveEnergy 증가량과도 일치한다.
    샘플이 부족한 시간은 비우고, 설비별로 시간 보간해 연속 시계열로 만든다.
    """
    h = pd.read_parquet(HOURLY_PATH)
    h = h[h["coverage"] >= MIN_COVERAGE]
    wide = h.pivot_table(index="dt_hour", columns="module", values="ap_mean", observed=True)
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    wide = wide.reindex(full_idx).interpolate(method="time", limit_direction="both")
    wide.index.name = "dt_hour"
    wide.columns = wide.columns.astype(str)
    return wide


def load_module_operation() -> pd.DataFrame:
    """설비×시간 operation 비율 (가동=1)"""
    h = pd.read_parquet(HOURLY_PATH)
    h = h[h["coverage"] >= MIN_COVERAGE]
    op = h.pivot_table(index="dt_hour", columns="module", values="op_ratio", observed=True)
    op.columns = op.columns.astype(str)
    return op


def plant_series(wide: pd.DataFrame) -> pd.Series:
    s = wide.sum(axis=1)
    s.name = "hourly_pow"
    return s


def is_holiday(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.normalize().isin(HOLIDAYS).astype(np.int8)


def target_index(month_start: str, hours: int = 672) -> pd.DatetimeIndex:
    return pd.date_range(month_start, periods=hours, freq="h")
