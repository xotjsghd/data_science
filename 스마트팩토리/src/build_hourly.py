"""
Step 1) 원본 RTU CSV(약 4.9GB)를 스트리밍으로 '한 번만' 읽어서
        설비(module) × 시간(hour) 단위 집계를 data/hourly.parquet 로 저장한다.
        이후 모든 실험은 이 parquet 파일만 사용한다.

집계 항목 (설비·시간별)
  - n / ts_nuniq           : 행 수 / 고유 timestamp 수 (5초 샘플링이면 정상 720)
  - ap_*                   : activePower 합계·평균·표준편차·최소·최대·0값 개수
  - op_ratio               : operation(가동=1) 비율
  - off_with_power         : operation=0 인데 activePower>0 인 행 수 (센서 이상 후보)
  - pcalc_mean             : Σ(상전압×상전류×역률/100) 평균 → activePower 단위 검증용
  - i_mean / q_mean        : 3상 전류 합 평균 / 무효전력 평균
  - ae_min / ae_max        : accumActiveEnergy(누적 전력량) 시간 내 최소·최대
  - lt_min / lt_max        : 그 시간에 관측된 첫/마지막 localtime

사용 예)
  python src/build_hourly.py                                  # RAW_DATA_DIR/rtu_data_full.csv
  python src/build_hourly.py --raw_csv "D:/data/rtu_data_full.csv"
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from data import RAW_DIR

MODULE_COL = "module(equipment)"
NUM_COLS = [
    "timestamp", "localtime", "operation",
    "voltageR", "voltageS", "voltageT",
    "currentR", "currentS", "currentT",
    "activePower",
    "powerFactorR", "powerFactorS", "powerFactorT",
    "reactivePowerLagging", "accumActiveEnergy",
]
INT_COLS = {"timestamp", "localtime", "operation", "accumActiveEnergy"}


def aggregate_batch(df: pd.DataFrame) -> pd.DataFrame:
    """한 배치를 (module, hour_key) 단위 부분 집계로 줄인다. hour_key = YYYYMMDDHH (int)."""
    df = df[df["localtime"].notna()]
    ap = df["activePower"]
    op = df["operation"]
    work = pd.DataFrame({
        "module": df[MODULE_COL].astype(str),
        "hk": (df["localtime"] // 10_000).astype("int64"),
        "ts": df["timestamp"],
        "lt": df["localtime"],
        "ap": ap,
        "ap_sq": ap * ap,
        "ap_zero": (ap == 0).astype("int32"),
        "op": op,
        "off_with_power": ((op == 0) & (ap > 0)).astype("int32"),
        "pcalc": (df["voltageR"] * df["currentR"] * df["powerFactorR"]
                  + df["voltageS"] * df["currentS"] * df["powerFactorS"]
                  + df["voltageT"] * df["currentT"] * df["powerFactorT"]) / 100.0,
        "isum": df["currentR"] + df["currentS"] + df["currentT"],
        "q": df["reactivePowerLagging"],
        "ae": df["accumActiveEnergy"],
    })
    g = work.groupby(["module", "hk"], sort=False)
    out = g.agg(
        n=("lt", "size"),
        ts_nuniq=("ts", "nunique"),
        ap_cnt=("ap", "count"),
        ap_sum=("ap", "sum"),
        ap_sqsum=("ap_sq", "sum"),
        ap_min=("ap", "min"),
        ap_max=("ap", "max"),
        ap_zero=("ap_zero", "sum"),
        op_sum=("op", "sum"),
        op_cnt=("op", "count"),
        off_with_power=("off_with_power", "sum"),
        pcalc_sum=("pcalc", "sum"),
        pcalc_cnt=("pcalc", "count"),
        isum_sum=("isum", "sum"),
        q_sum=("q", "sum"),
        q_cnt=("q", "count"),
        ae_min=("ae", "min"),
        ae_max=("ae", "max"),
        ae_cnt=("ae", "count"),
        lt_min=("lt", "min"),
        lt_max=("lt", "max"),
    )
    return out


SUM_KEYS = ["n", "ts_nuniq", "ap_cnt", "ap_sum", "ap_sqsum", "ap_zero", "op_sum", "op_cnt",
            "off_with_power", "pcalc_sum", "pcalc_cnt", "isum_sum", "q_sum", "q_cnt", "ae_cnt"]
MIN_KEYS = ["ap_min", "ae_min", "lt_min"]
MAX_KEYS = ["ap_max", "ae_max", "lt_max"]


def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """배치 경계에 걸친 (module, hour)를 다시 합친다."""
    allp = pd.concat(parts)
    g = allp.groupby(level=["module", "hk"])
    out = pd.concat([g[SUM_KEYS].sum(), g[MIN_KEYS].min(), g[MAX_KEYS].max()], axis=1)
    return out


def finalize(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.reset_index()
    df["dt_hour"] = pd.to_datetime(df["hk"].astype(str), format="%Y%m%d%H")
    df["ap_mean"] = df["ap_sum"] / df["ap_cnt"].replace(0, np.nan)
    var = df["ap_sqsum"] / df["ap_cnt"].replace(0, np.nan) - df["ap_mean"] ** 2
    df["ap_std"] = np.sqrt(var.clip(lower=0))
    df["op_ratio"] = df["op_sum"] / df["op_cnt"].replace(0, np.nan)
    df["pcalc_mean"] = df["pcalc_sum"] / df["pcalc_cnt"].replace(0, np.nan)
    df["i_mean"] = df["isum_sum"] / df["pcalc_cnt"].replace(0, np.nan)
    df["q_mean"] = df["q_sum"] / df["q_cnt"].replace(0, np.nan)
    df["coverage"] = df["ts_nuniq"] / 720.0  # 5초 간격 기준 1시간 = 720샘플
    df["module"] = df["module"].astype("category")
    cols = ["module", "dt_hour", "n", "ts_nuniq", "coverage",
            "ap_cnt", "ap_sum", "ap_mean", "ap_std", "ap_min", "ap_max", "ap_zero",
            "op_ratio", "off_with_power", "pcalc_mean", "i_mean", "q_mean",
            "ae_min", "ae_max", "ae_cnt", "lt_min", "lt_max"]
    return df[cols].sort_values(["module", "dt_hour"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", default=str(RAW_DIR / "rtu_data_full.csv"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "hourly.parquet"))
    ap.add_argument("--profile_out", default=str(Path(__file__).resolve().parents[1] / "reports" / "raw_profile.json"))
    ap.add_argument("--block_mb", type=int, default=128)
    args = ap.parse_args()

    col_types = {c: (pa.int64() if c in INT_COLS else pa.float64()) for c in NUM_COLS}
    col_types[MODULE_COL] = pa.string()
    reader = pacsv.open_csv(
        args.raw_csv,
        read_options=pacsv.ReadOptions(block_size=args.block_mb << 20),
        convert_options=pacsv.ConvertOptions(include_columns=[MODULE_COL] + NUM_COLS,
                                             column_types=col_types),
    )

    t0 = time.time()
    parts, total_rows = [], 0
    null_counts = {c: 0 for c in [MODULE_COL] + NUM_COLS}
    neg_ap = 0
    for i, batch in enumerate(reader, start=1):
        for c in null_counts:
            null_counts[c] += batch.column(c).null_count
        df = batch.to_pandas()
        neg_ap += int((df["activePower"] < 0).sum())
        total_rows += len(df)
        parts.append(aggregate_batch(df))
        if i % 10 == 0:
            print(f"[batch {i}] rows={total_rows:,}  elapsed={time.time() - t0:.0f}s", flush=True)
            parts = [combine(parts)]  # 메모리 절약을 위해 중간 병합

    hourly = finalize(combine(parts))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.out, index=False)

    profile = {
        "source": Path(args.raw_csv).name,
        "total_rows": total_rows,
        "null_counts": null_counts,
        "negative_activePower_rows": neg_ap,
        "modules": hourly["module"].cat.categories.tolist(),
        "dt_min": str(hourly["dt_hour"].min()),
        "dt_max": str(hourly["dt_hour"].max()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(args.profile_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.profile_out).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out}  shape={hourly.shape}")


if __name__ == "__main__":
    main()
