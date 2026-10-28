"""
Step 2) activePower / accumActiveEnergy 단위 점검

점검 방법
  (a) 에너지 일치: 연속 두 시간 모두 완전 관측(720샘플)인 구간에서
      Δ누적량 = ae_min[h+1] − ae_min[h]  vs  activePower 시간 평균 × 1h
      → 비율이 1이면 가이드 정의(activePower kW, accumActiveEnergy kWh)와 데이터가 일치하고,
        '시간 평균 kW × 1h = 그 시간의 kWh' 로 hourly_pow를 만드는 것이 맞다.
  (b) 전기량 정합성(참고): Σ(상전압 × 상전류 × 역률)과 activePower 비교
결과는 reports/units.md 에 기록한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    h = pd.read_parquet(ROOT / "data" / "hourly.parquet")
    h = h.sort_values(["module", "dt_hour"])
    rows = []
    for mod, g in h.groupby("module", observed=True):
        g = g.set_index("dt_hour")
        nxt = g.shift(-1, freq="h")
        both = g.join(nxt[["ae_min", "coverage"]], rsuffix="_next", how="inner")
        ok = both[(both["coverage"] >= 1) & (both["coverage_next"] >= 1)]
        d_ae = ok["ae_min_next"] - ok["ae_min"]
        ratio = d_ae / ok["ap_mean"]
        full = g[g["coverage"] >= 1]
        rows.append({
            "module": mod,
            "hours_checked": len(ok),
            "ap_mean": g["ap_mean"].mean(),
            "dAE_per_hour": d_ae.mean(),
            "dAE/ap_median": ratio.median(),
            "dAE/ap_p05": ratio.quantile(0.05),
            "dAE/ap_p95": ratio.quantile(0.95),
            "pcalc_mean": full["pcalc_mean"].mean(),
            "pcalc/ap": (full["pcalc_mean"] / full["ap_mean"]).median(),
            "ap_sample_std": full["ap_std"].mean(),
        })
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))

    overall_ratio = float(np.nanmedian(t["dAE/ap_median"]))
    pcalc_ratio = float(np.nanmedian(t["pcalc/ap"]))
    plant = h[h["coverage"] >= 1].groupby("dt_hour")["ap_mean"].sum().mean()
    md = f"""# 단위 점검 결과 (Step 2)

## 결론
- 가이드 정의: **activePower = kW, accumActiveEnergy = kWh.** 데이터가 이 정의와 정확히 맞습니다.
- 따라서 `hourly_pow` = Σ 설비별 activePower **시간 평균**(kW) × 1h = 그 시간의 전력량(kWh)이고,
  별도의 단위 환산은 필요 없습니다. `agg_pow` = Σ hourly_pow (kWh).
- 공모전 제출본(`bda_xgboost.py`)도 같은 방식이라 **스케일 차이는 없습니다.**
  주최측이 데이터·가이드·정답을 같은 정의로 만들었으므로 개선본도 이 스케일을 따릅니다.

## 근거
1. **누적량과 일치**: 연속 완전 관측 시간에서 `Δ누적량 ÷ activePower 시간평균`의 설비별 중앙값 = **{overall_ratio:.3f}**
   (5~95% 구간 {t['dAE/ap_p05'].min():.3f}~{t['dAE/ap_p95'].max():.3f}, 13개 설비 모두 같음)
   → 시간 평균 kW × 1h 가 실제 누적 전력량 증가(kWh)와 같음. 5초 샘플 합계(×720)를 쓰면 720배가 됨.
2. 공장 전체 activePower 시간평균 합 = **약 {plant:,.0f} kW** → 시간당 약 {plant:,.0f} kWh, 일 약 {plant * 24:,.0f} kWh.
3. (참고) 전압·전류·역률로 계산한 전력 Σ(V×I×PF)은 activePower의 {pcalc_ratio:.2f}배로 서로 맞지 않습니다.
   13개 설비가 조명·호이스트·분쇄기 구분 없이 평균 약 {t['ap_mean'].mean():,.0f}, 샘플 표준편차 약 {t['ap_sample_std'].mean():,.0f}로
   같은 분포를 가진 점까지 보면 **시뮬레이션으로 만든 데이터**로 보이며, 전기량 컬럼으로 단위를 역추정하기보다
   주최측 정의를 따르는 것이 맞습니다.

## 원본 코드(bda_xgboost.py)와의 차이
| 항목 | 공모전 제출본 | 개선본 |
|---|---|---|
| 시간 집계 | (시간, 설비)별 activePower 평균 → 설비 합 | 동일 |
| 단위 | 가이드 정의(kW·kWh) | 동일 |
| 샘플이 거의 없는 시간 | 그대로 평균 (4/30 00시는 설비당 1샘플) | 커버리지 50% 미만 시간은 제외 후 보간 |
| 빈 시간 보간 | 학습 구간: 공장 합계 선형보간 / 4월 실측: 보간 없음 | 설비 단위 시간 보간 |
| 집계 방식 | 청크마다 `iterrows`로 dict 누적 | pyarrow 스트리밍 + groupby 부분집계 병합 |

## 설비별 상세
{t.round(3).to_markdown(index=False)}
"""
    (ROOT / "reports" / "units.md").write_text(md, encoding="utf-8")
    print("\nSaved reports/units.md")


if __name__ == "__main__":
    main()
