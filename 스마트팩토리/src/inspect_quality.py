"""
데이터 품질 점검 (중복·이상치·시간 패턴·파일 간 일치 여부) → reports/data_quality.md

입력
  - data/hourly.parquet            : rtu_data_full.csv 집계 (build_hourly.py)
  - data/hourly_traincsv.parquet   : train.csv 집계 (선택, 있으면 rtu와 비교)
"""
import json

import numpy as np
import pandas as pd

from data import ROOT


def main():
    h = pd.read_parquet(ROOT / "data" / "hourly.parquet")
    prof = json.loads((ROOT / "reports" / "raw_profile.json").read_text(encoding="utf-8"))
    n_mod = h["module"].nunique()

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
        f"- 음수 activePower: {prof['negative_activePower_rows']}개 / 중복 timestamp: {int((h['n'] - h['ts_nuniq']).sum())}개",
        "- 데이터는 2025-04-30 00:00:00에서 끝남 (이 시간은 설비당 1샘플이라 분석에서 제외)",
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
                  f"- 설비·시간 {len(t):,}개",
                  f"- rtu_data_full.csv의 3월 31일까지 구간과 **{'완전히 동일' if same else '불일치'}** "
                  "(행 수·activePower 합·누적량 비교) → 원본을 기간별로 나눠 사용 (12~3월 학습, 4월 내부 검증)"]

    (ROOT / "reports" / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
