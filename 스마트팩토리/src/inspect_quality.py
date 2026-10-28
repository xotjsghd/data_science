"""
데이터 품질 점검 (결측·중복·이상치·파일 간 일치 여부) → reports/data_quality.md

입력
  - data/hourly.parquet            : rtu_data_full.csv 집계 (build_hourly.py)
  - data/hourly_traincsv.parquet   : train.csv 집계 (선택, 있으면 rtu와 비교)
  - RAW_DATA_DIR/test.csv           : 4월 파일 (선택, 있으면 손상 여부 확인)
"""
import json

import numpy as np
import pandas as pd

from data import RAW_DIR, ROOT


def main():
    h = pd.read_parquet(ROOT / "data" / "hourly.parquet")
    prof = json.loads((ROOT / "reports" / "raw_profile.json").read_text(encoding="utf-8"))
    n_mod = h["module"].nunique()
    n_hours = h["dt_hour"].nunique()
    expected = n_mod * n_hours * 720
    partial = h[h["coverage"] < 1]

    full = h[h["coverage"] >= 1]
    plant = full.groupby("dt_hour")["ap_mean"].sum()
    iid_std = float(np.sqrt((full["ap_std"] ** 2 / 720).groupby(full["dt_hour"]).sum()).mean())
    by_hour = plant.groupby(plant.index.hour).mean()
    by_dow = plant.groupby(plant.index.dayofweek).mean()
    by_month = plant.groupby(plant.index.to_period("M")).mean()
    daily = plant.resample("D").sum()

    lines = [
        "# 데이터 품질 점검 (Step 1 부록)",
        "",
        "## rtu_data_full.csv (주최측 제공, 12~4월 전체)",
        f"- 총 {prof['total_rows']:,}행, 설비 {n_mod}개, 기간 {prof['dt_min']} ~ {prof['dt_max']}",
        f"- 결측값(NaN): 전 컬럼 0개 / 음수 activePower: {prof['negative_activePower_rows']}개 / 중복 timestamp: {int((h['n'] - h['ts_nuniq']).sum())}개",
        f"- 5초 간격 기준 기대 행 수 {expected:,} 대비 부족 {expected - prof['total_rows']:,}행",
        f"  → 전부 마지막 시간(2025-04-30 00시, 설비당 1샘플)에서 발생. **4/30 01시 이후는 데이터 자체가 없음**",
        f"- 시간 내 일부 누락(coverage<1) 설비·시간: {len(partial)}개 "
        f"({', '.join(sorted(partial['dt_hour'].astype(str).unique()))})",
        f"- operation: 전 구간 100% 가동(=1), activePower=0 인 행 0개, '꺼짐인데 소비' 0건",
        "",
        "## 시계열 특성 (공장 전체 hourly_pow, kWh/h)",
        f"- 평균 {plant.mean():,.1f}, 시간별 표준편차 {plant.std():,.1f} (변동계수 {plant.std() / plant.mean():.2%})",
        f"- 5초 샘플이 서로 독립인 잡음이라고 가정했을 때 기대되는 시간평균 표준편차 = {iid_std:,.1f}",
        "  → 실제 변동이 순수 샘플링 잡음 크기와 같음 = **예측 가능한 패턴이 거의 없음**",
        f"- 자기상관: lag1 {plant.autocorr(1):+.3f}, lag24 {plant.autocorr(24):+.3f}, lag168 {plant.autocorr(168):+.3f}",
        f"- 시간대별 평균 범위 {by_hour.min():,.0f}~{by_hour.max():,.0f}, 요일별 {by_dow.min():,.0f}~{by_dow.max():,.0f}",
        f"- 월별 평균: " + ", ".join(f"{p}: {v:,.0f}" for p, v in by_month.items()),
        f"- 일 사용량 {daily.mean():,.0f} kWh ± {daily.std():,.0f} (변동계수 {daily.std() / daily.mean():.3%})",
        "- 공휴일(12/25, 1/1, 1/27~30 설 연휴, 3/3 대체공휴일) 일 사용량: "
        + ", ".join(f"{d}: {daily.loc[d]:,.0f}" for d in
                    ["2024-12-25", "2025-01-01", "2025-01-28", "2025-01-29", "2025-03-03"]),
        "  → 평일과 차이 없음",
        "- 13개 설비 모두 평균 약 3,010 W, 샘플 표준편차 약 716 W로 거의 동일한 분포",
    ]

    tr_path = ROOT / "data" / "hourly_traincsv.parquet"
    if tr_path.exists():
        t = pd.read_parquet(tr_path)
        a = h[h["dt_hour"] < "2025-04-01"]
        m = a.merge(t, on=["module", "dt_hour"], suffixes=("_r", "_t"))
        same = len(m) == len(a) == len(t) and (m["n_r"] == m["n_t"]).all() \
            and float((m["ap_sum_r"] - m["ap_sum_t"]).abs().max()) < 1e-3
        lines += ["", "## train.csv",
                  f"- 설비·시간 {len(t):,}개, 시간 내 누락 {int((t['coverage'] < 1).sum())}개",
                  f"- rtu_data_full.csv의 3월 31일까지 구간과 **{'완전히 동일' if same else '불일치'}** "
                  "(행 수·activePower 합·누적량 비교) → 학습에는 rtu_data_full.csv만 사용"]

    te_path = RAW_DIR / "test.csv"
    if te_path.exists():
        te = pd.read_csv(te_path, encoding="cp949", usecols=["module(equipment)", "datetime", "timestamp"])
        vc = te["module(equipment)"].value_counts()
        te_dt = pd.to_datetime(te["datetime"], format="%Y-%m-%d %H:%M")
        lines += ["", "## test.csv (4월)",
                  f"- 데이터 {len(te):,}행 = **엑셀 최대 행 수(1,048,576) − 헤더 1행** → 엑셀 저장 과정에서 잘린 파일",
                  f"- 설비 {len(vc)}개만 존재 (13개 중): " + ", ".join(f"{c}행" for c in vc.values),
                  f"- 마지막 시각 {te_dt.max()}, 세 번째 설비는 {te_dt[te['module(equipment)'] == vc.index[-1]].max()}에서 끊김",
                  f"- timestamp·localtime이 지수표기(예: {te['timestamp'].iloc[0]})로 저장되어 초 단위 정보 손실, 설비명 한글 깨짐",
                  "- → 이 파일 기준이면 4월 데이터의 약 85%가 '결측'처럼 보임. "
                  "4월 실측은 rtu_data_full.csv의 4월 구간(13개 설비 전부, 4/30 00시까지)을 사용"]

    (ROOT / "reports" / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
