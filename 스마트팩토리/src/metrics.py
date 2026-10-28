"""
Step 3) 대회 평가 지표 구현 (메타데이터 가이드 v1 기준)

| 항목        | 입력                   | 처리                       | 지표                         | 비중 |
|-------------|------------------------|----------------------------|------------------------------|------|
| Daily       | hourly_pow (672개)     | 일 단위 합산               | MAE, RMSE, SMAPE → 1/(1+오차) | 30%  |
| Weekly      | hourly_pow (672개)     | 주 단위 합산(월요일 시작)  | MAE, RMSE, SMAPE → 1/(1+오차) | 30%  |
| may_bill    | agg_pow × 180원/kWh    | 상대오차                   | Bell-curve                   | 20%  |
| may_carbon  | agg_pow × 0.424kg/kWh  | 상대오차                   | Bell-curve                   | 20%  |

가이드에 명시되지 않아 가정한 부분 (README에도 기록)
  1) Bell-curve: "baseline 10%, std 5% → 오차 10%=0.5점, 20%=0.022점, 30%↑≈0점" 예시와
     정확히 일치하는 식은 정규분포 생존함수  score = 1 − Φ((|상대오차| − 0.10) / 0.05).
  2) Daily/Weekly 점수: MAE·RMSE 원값(kWh)에 1/(1+오차)를 그대로 쓰면 일 합산 오차가 수백 kWh라
     모든 모델 점수가 0 근처로 붙어 비교가 안 된다. 그래서 MAE·RMSE는 실측 평균으로 나눈 정규화 값(NMAE, NRMSE),
     SMAPE는 0~2 비율로 두고 각각 1/(1+오차) 변환 후 평균한 값을 '점수 근사치'로 사용한다.
     원값(MAE/RMSE kWh, SMAPE %)도 함께 보고한다.
  3) 요금·탄소의 상대오차는 둘 다 agg_pow 상대오차와 같으므로 사실상 agg_pow 정확도가 40% 비중.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BILL_RATE = 180.0      # 원/kWh
CARBON_RATE = 0.424    # kgCO2/kWh
BELL_BASELINE = 0.10
BELL_STD = 0.05
WEIGHTS = {"daily": 0.3, "weekly": 0.3, "bill": 0.2, "carbon": 0.2}


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def smape(y, p, eps=1e-9) -> float:
    """0~2 비율 (×100 하면 %)"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2.0 * np.abs(p - y) / np.maximum(np.abs(y) + np.abs(p), eps)))


def bell_score(rel_err: float, baseline=BELL_BASELINE, std=BELL_STD) -> float:
    """1 − Φ((|e| − baseline)/std). e=0 → 0.977, e=10% → 0.5, e=20% → 0.023."""
    z = (abs(rel_err) - baseline) / std
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def daily_sum(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.floor("D")).sum()


def weekly_sum(s: pd.Series) -> pd.Series:
    """월요일 시작 주 단위 합산 (기간 양 끝의 부분 주도 하나의 주로 취급)"""
    week_start = (s.index.floor("D") - pd.to_timedelta(s.index.dayofweek, unit="D"))
    return s.groupby(week_start).sum()


def _agg_block(y: pd.Series, p: pd.Series) -> dict:
    m_mae, m_rmse, m_smape = mae(y, p), rmse(y, p), smape(y, p)
    mean_y = float(np.mean(y))
    nmae, nrmse = m_mae / mean_y, m_rmse / mean_y
    score = np.mean([1 / (1 + nmae), 1 / (1 + nrmse), 1 / (1 + m_smape)])
    return {"mae": m_mae, "rmse": m_rmse, "smape_pct": m_smape * 100,
            "nmae_pct": nmae * 100, "nrmse_pct": nrmse * 100, "score": float(score)}


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    y_true, y_pred: DatetimeIndex(시간 단위)를 가진 hourly_pow(kWh) 시리즈. 같은 672시간을 가정.
    반환: 원지표 + 항목별 점수 + 가중 합산 점수(score_total, 0~1)
    """
    idx = y_true.index.intersection(y_pred.index)
    if len(idx) != len(y_true):
        raise ValueError(f"예측 시간축 불일치: true={len(y_true)}, 공통={len(idx)}")
    y, p = y_true.loc[idx].astype(float), y_pred.loc[idx].astype(float)

    d = _agg_block(daily_sum(y), daily_sum(p))
    w = _agg_block(weekly_sum(y), weekly_sum(p))
    agg_true, agg_pred = float(y.sum()), float(p.sum())
    rel = (agg_pred - agg_true) / agg_true
    bill = bell_score(rel)      # bill = agg_pow×180 → 상대오차 동일
    carbon = bell_score(rel)    # carbon = agg_pow×0.424 → 상대오차 동일

    total = (WEIGHTS["daily"] * d["score"] + WEIGHTS["weekly"] * w["score"]
             + WEIGHTS["bill"] * bill + WEIGHTS["carbon"] * carbon)
    return {
        "hourly_mae": mae(y, p), "hourly_smape_pct": smape(y, p) * 100,
        **{f"daily_{k}": v for k, v in d.items()},
        **{f"weekly_{k}": v for k, v in w.items()},
        "agg_true_kwh": agg_true, "agg_pred_kwh": agg_pred, "agg_rel_err_pct": rel * 100,
        "bill_score": bill, "carbon_score": carbon,
        "score_total": float(total),
    }


def make_submission(pred: pd.Series) -> pd.DataFrame:
    """제출 포맷: id, hourly_pow, agg_pow, may_bill, may_carbon (스칼라 값은 모든 행 반복)"""
    agg = float(pred.sum())
    return pd.DataFrame({
        "id": pred.index.strftime("%Y-%m-%d %H:%M:%S"),
        "hourly_pow": pred.values.astype(float),
        "agg_pow": agg,
        "may_bill": agg * BILL_RATE,
        "may_carbon": agg * CARBON_RATE,
    })


if __name__ == "__main__":
    # 가이드 예시와 Bell-curve 식이 일치하는지 확인
    for e in [0.0, 0.05, 0.10, 0.20, 0.30]:
        print(f"rel_err={e:.0%}  bell={bell_score(e):.4f}")
    idx = pd.date_range("2025-05-01", periods=672, freq="h")
    y = pd.Series(100.0, index=idx)
    print("완전 일치:", round(evaluate(y, y)["score_total"], 4))
    print("+10% 일괄:", {k: round(v, 4) for k, v in evaluate(y, y * 1.1).items()
                        if k in ("daily_score", "weekly_score", "bill_score", "score_total")})
    print("주 단위 구간:", weekly_sum(y).index.strftime("%m-%d").tolist(), weekly_sum(y).values.tolist())
