"""
Step 5) 비교 모델 모음. 모든 모델은 같은 인터페이스를 따른다.

    fn(hist, op, target) -> pd.Series (kWh, index=target)
      hist   : 예측 시작 이전의 설비별 시간 전력량(kWh) wide 테이블 (index=dt_hour, columns=module)
      op     : 같은 모양의 operation 비율 테이블
      target : 예측할 672개 시간 (DatetimeIndex)

모델 분류
  - orig_*      : 공모전 제출본 방식 재현 (XGBoost + lag_1 재귀 예측)
  - base_*      : 단순 기준 모델 (평균/요일×시간 평균)
  - *_direct*   : lag ≥ 672시간만 쓰는 직접 예측 모델 (재귀 없음)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from data import is_holiday, plant_series

H = pd.Timedelta(hours=1)
WEEK4 = 672


# ----------------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------------
def dow_hour_profile(s: pd.Series) -> pd.Series:
    return s.groupby([s.index.dayofweek, s.index.hour]).mean()


def map_profile(prof: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    key = pd.MultiIndex.from_arrays([idx.dayofweek, idx.hour])
    return prof.reindex(key).to_numpy()


def fill_gap(s: pd.Series, start: pd.Timestamp) -> pd.Series:
    """마지막 관측 시각과 예측 시작 사이의 빈 시간(예: 4/30 01~23시)을 최근 4주 요일×시간 평균으로 채운다."""
    last = s.index.max()
    if last >= start - H:
        return s
    gap = pd.date_range(last + H, start - H, freq="h")
    fill = pd.Series(map_profile(dow_hour_profile(s.iloc[-WEEK4:]), gap), index=gap)
    return pd.concat([s, fill])


# ----------------------------------------------------------------------------
# 1) 공모전 제출본 재현: XGBoost + lag_1 재귀 예측
#    - 공장 전체 합계 시리즈로 학습, 마지막 7일은 학습에서 제외(원본 그대로)
#    - 미래 672시간을 한 시간씩 예측값을 다시 lag_1 입력으로 넣어가며 예측
# ----------------------------------------------------------------------------
ORIG_FEATS = ["hour", "dow", "hour_sin", "hour_cos", "lag_1", "lag_24", "lag_168", "roll_24", "roll_168"]
ORIG_PARAMS = dict(n_estimators=1800, learning_rate=0.02, max_depth=6, subsample=0.7, min_child_weight=5,
                   colsample_bytree=0.7, reg_lambda=1.0, objective="reg:squarederror", tree_method="hist",
                   random_state=42, n_jobs=-1)


def _orig_frame(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s.astype(np.float32)})
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    for l in (1, 24, 168):
        df[f"lag_{l}"] = df["y"].shift(l)
    df["roll_24"] = df["y"].shift(1).rolling(24).mean()
    df["roll_168"] = df["y"].shift(1).rolling(168).mean()
    return df.dropna().astype(float)


def orig_xgb_recursive(hist, op, target):
    s = plant_series(hist)
    df = _orig_frame(s)
    train = df[df.index <= df.index.max() - pd.Timedelta(days=7)]
    model = XGBRegressor(**ORIG_PARAMS).fit(train[ORIG_FEATS], train["y"])
    booster = model.get_booster()

    buf = list(s.to_numpy(dtype=float))
    preds = []
    for t in target:
        h = t.hour
        x = [h, t.dayofweek, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
             buf[-1], buf[-24], buf[-168], np.mean(buf[-24:]), np.mean(buf[-168:])]
        p = max(float(booster.inplace_predict(np.asarray([x], dtype=float))[0]), 0.0)
        preds.append(p)
        buf.append(p)
    return pd.Series(preds, index=target)


# ----------------------------------------------------------------------------
# 2) 단순 기준 모델
# ----------------------------------------------------------------------------
def base_dowhour_4w(hist, op, target):
    """최근 4주 요일×시간 평균"""
    s = plant_series(hist).iloc[-WEEK4:]
    return pd.Series(map_profile(dow_hour_profile(s), target), index=target)


def base_mean_4w(hist, op, target):
    """최근 4주 전체 평균(상수)"""
    return pd.Series(plant_series(hist).iloc[-WEEK4:].mean(), index=target)


def base_mean_all(hist, op, target):
    """학습 기간 전체 평균(상수)"""
    return pd.Series(plant_series(hist).mean(), index=target)


def base_mean_all_hol(hist, op, target):
    """전체 평균 + 과거 휴일/평일 비율을 예측 기간 휴일에 적용"""
    s = plant_series(hist)
    hol = is_holiday(s.index).astype(bool)
    normal = s[~hol].mean()
    ratio = s[hol].mean() / normal if hol.any() else 1.0
    return pd.Series(np.where(is_holiday(target), normal * ratio, normal), index=target)


# ----------------------------------------------------------------------------
# 3) 직접 예측 (lag ≥ 672) — 재귀 없이 예측 시점에 이미 알려진 값만 사용
# ----------------------------------------------------------------------------
DIRECT_LAGS = (672, 840, 1008)  # 4·5·6주 전 같은 요일·시간


def _direct_features(s: pd.Series, idx: pd.DatetimeIndex, op_s: pd.Series | None = None,
                     holiday: bool = False) -> pd.DataFrame:
    full_idx = pd.date_range(s.index.min(), max(s.index.max(), idx.max()), freq="h")
    ext = s.reindex(full_idx)
    X = pd.DataFrame(index=full_idx)
    X["hour"] = full_idx.hour
    X["dow"] = full_idx.dayofweek
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    for L in DIRECT_LAGS:
        X[f"lag_{L}"] = ext.shift(L)
    X["lag_how_mean"] = X[[f"lag_{L}" for L in DIRECT_LAGS]].mean(axis=1)
    X["roll24_at672"] = ext.shift(672).rolling(24).mean()
    X["roll168_at672"] = ext.shift(672).rolling(168).mean()
    if holiday:
        X["holiday"] = is_holiday(full_idx)
    if op_s is not None:
        op_ext = op_s.reindex(full_idx).ffill()
        for L in DIRECT_LAGS:
            X[f"op_lag_{L}"] = op_ext.shift(L)
    return X.loc[idx]


def _make_gbm(kind: str):
    if kind == "xgb":
        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                            random_state=42, n_jobs=8)
    return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=42, n_jobs=8, verbose=-1)


def _direct_plant(hist, op, target, kind="lgbm", holiday=False, use_op=False):
    s = fill_gap(plant_series(hist), target[0])
    op_s = op.mean(axis=1) if use_op else None
    X_all = _direct_features(s, s.index, op_s, holiday)
    ok = X_all.notna().all(axis=1)
    model = _make_gbm(kind).fit(X_all[ok], s[ok])
    X_t = _direct_features(s, target, op_s, holiday)
    return pd.Series(model.predict(X_t), index=target)


def xgb_direct(hist, op, target):
    return _direct_plant(hist, op, target, kind="xgb")


def lgbm_direct(hist, op, target):
    return _direct_plant(hist, op, target, kind="lgbm")


def lgbm_direct_hol(hist, op, target):
    return _direct_plant(hist, op, target, kind="lgbm", holiday=True)


def lgbm_direct_op(hist, op, target):
    return _direct_plant(hist, op, target, kind="lgbm", use_op=True)


def lgbm_direct_module(hist, op, target):
    """설비별 패턴: 설비 13개를 한 모델로 학습(설비 ID를 범주형 피처로) → 설비별 예측 합산"""
    Xs, ys, Xts = [], [], []
    for code, m in enumerate(hist.columns):
        s = fill_gap(hist[m], target[0])
        X = _direct_features(s, s.index, holiday=True)
        ok = X.notna().all(axis=1)
        X = X[ok].assign(module=code)
        Xs.append(X)
        ys.append(s[ok])
        Xts.append(_direct_features(s, target, holiday=True).assign(module=code))
    X, y, Xt = pd.concat(Xs), pd.concat(ys), pd.concat(Xts)
    X["module"] = X["module"].astype("category")
    Xt["module"] = pd.Categorical(Xt["module"], categories=X["module"].cat.categories)
    model = _make_gbm("lgbm").fit(X, y)
    pred = pd.Series(model.predict(Xt), index=Xt.index)
    return pred.groupby(level=0).sum().reindex(target)


MODELS = {
    "orig_xgb_recursive": orig_xgb_recursive,
    "base_dowhour_4w": base_dowhour_4w,
    "base_mean_4w": base_mean_4w,
    "base_mean_all": base_mean_all,
    "base_mean_all_hol": base_mean_all_hol,
    "xgb_direct": xgb_direct,
    "lgbm_direct": lgbm_direct,
    "lgbm_direct_hol": lgbm_direct_hol,
    "lgbm_direct_op": lgbm_direct_op,
    "lgbm_direct_module": lgbm_direct_module,
}

DESCRIPTIONS = {
    "orig_xgb_recursive": "공모전 제출본 재현: XGBoost(lag_1/24/168, roll) 672시간 재귀 예측",
    "base_dowhour_4w": "최근 4주 요일×시간 평균",
    "base_mean_4w": "최근 4주 평균 (상수)",
    "base_mean_all": "학습 전체 기간 평균 (상수)",
    "base_mean_all_hol": "전체 평균 + 과거 휴일 비율 보정",
    "xgb_direct": "XGBoost 직접 예측 (lag 672/840/1008 + 달력)",
    "lgbm_direct": "LightGBM 직접 예측 (lag 672/840/1008 + 달력)",
    "lgbm_direct_hol": "LightGBM 직접 예측 + 공휴일 피처",
    "lgbm_direct_op": "LightGBM 직접 예측 + operation 비율 lag",
    "lgbm_direct_module": "LightGBM 설비별 학습 후 합산 (+공휴일)",
    "oracle_flat": "(참고) 정답 월평균을 상수로 예측 — 잡음 때문에 넘을 수 없는 하한",
}
