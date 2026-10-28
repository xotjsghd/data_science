import os
import re
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# -------------------------
# Utils
# -------------------------
def clean_module_to_id(s: str) -> str:
    if pd.isna(s):
        return "UNKNOWN"
    s = str(s).strip()
    s = re.sub(r"[()\-]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "UNKNOWN"


def parse_localtime(series: pd.Series) -> pd.Series:
    """
    localtime: YYYYMMDDHHMMSS (int) -> datetime
    """
    s = series.astype("int64").astype(str).str.zfill(14)
    return pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")


# -------------------------
# Step 1) Build hourly plant series from big CSV
#   - train_hist: dt < 2025-04-01 (4월 배제)
#   - april_true: 2025-04-01 ~ 2025-04-30 23:00 (검증용 실제값)
#
# ✅ FIX:
#   - train_hist는 연속성 확보 위해 interpolate OK
#   - april_true는 "실제 관측된 시간"만 남김 (reindex/interpolate 금지)
# -------------------------
def build_hourly_series_split_april(
    csv_path: str,
    chunksize: int,
    april_start="2025-04-01 00:00:00",
    april_end="2025-04-30 23:00:00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")

    april_start = pd.Timestamp(april_start)
    april_end = pd.Timestamp(april_end)

    # 누적용 dict (time,module)->sum,count
    sum_train, cnt_train = {}, {}
    sum_apr, cnt_apr = {}, {}

    reader = pd.read_csv(
        csv_path,
        usecols=["localtime", "module(equipment)", "activePower"],
        chunksize=chunksize,
        dtype={
            "localtime": "int64",
            "module(equipment)": "category",
            "activePower": "float32",
        },
    )

    for i, chunk in enumerate(reader, start=1):
        dt = parse_localtime(chunk["localtime"])
        chunk = chunk.loc[dt.notna()].copy()
        chunk["dt"] = dt.loc[dt.notna()].values

        chunk = chunk.dropna(subset=["activePower"])
        chunk["dt_hour"] = chunk["dt"].dt.floor("h")
        chunk["module_id"] = chunk["module(equipment)"].map(clean_module_to_id)

        # train 부분: 4월 시작 이전
        m_train = chunk["dt_hour"] < april_start
        if m_train.any():
            g = chunk.loc[m_train].groupby(
                ["dt_hour", "module_id"], observed=True
            )["activePower"].agg(["sum", "count"])
            for (dt_hour, module_id), row in g.iterrows():
                key = (dt_hour, module_id)
                sum_train[key] = sum_train.get(key, 0.0) + float(row["sum"])
                cnt_train[key] = cnt_train.get(key, 0) + int(row["count"])

        # april 실제: 4월 한달
        m_apr = (chunk["dt_hour"] >= april_start) & (chunk["dt_hour"] <= april_end)
        if m_apr.any():
            g = chunk.loc[m_apr].groupby(
                ["dt_hour", "module_id"], observed=True
            )["activePower"].agg(["sum", "count"])
            for (dt_hour, module_id), row in g.iterrows():
                key = (dt_hour, module_id)
                sum_apr[key] = sum_apr.get(key, 0.0) + float(row["sum"])
                cnt_apr[key] = cnt_apr.get(key, 0) + int(row["count"])

        print(f"[chunk {i}] processed rows={len(chunk):,}")

    def to_plant_hourly(sum_dict, cnt_dict, do_interpolate: bool) -> pd.DataFrame:
        """
        (time,module) mean -> sum over modules => plant hourly series
        do_interpolate=True  : 연속 인덱스 생성 + 선형 보간 (train_hist 용)
        do_interpolate=False : 관측된 시간만 유지 (april_true 용)
        """
        rows = []
        for (dt_hour, module_id), s in sum_dict.items():
            c = cnt_dict[(dt_hour, module_id)]
            rows.append((dt_hour, s / max(c, 1)))

        if not rows:
            return pd.DataFrame(columns=["dt_hour", "hourly_pow"])

        df_mod_hour = pd.DataFrame(rows, columns=["dt_hour", "ap_mean"])
        df_hour = (
            df_mod_hour.groupby("dt_hour", as_index=False)["ap_mean"]
            .sum()
            .rename(columns={"ap_mean": "hourly_pow"})
            .sort_values("dt_hour")
            .reset_index(drop=True)
        )

        if do_interpolate:
            full_idx = pd.date_range(df_hour["dt_hour"].min(), df_hour["dt_hour"].max(), freq="h")
            df_hour = df_hour.set_index("dt_hour").reindex(full_idx)
            df_hour["hourly_pow"] = df_hour["hourly_pow"].interpolate(
                method="linear", limit_direction="both"
            )
            df_hour = df_hour.reset_index().rename(columns={"index": "dt_hour"})

        return df_hour

    df_train_hist = to_plant_hourly(sum_train, cnt_train, do_interpolate=True)
    df_april_true = to_plant_hourly(sum_apr, cnt_apr, do_interpolate=False)

    if not df_april_true.empty:
        df_april_true = df_april_true[
            (df_april_true["dt_hour"] >= april_start) & (df_april_true["dt_hour"] <= april_end)
        ].copy()

    return df_train_hist, df_april_true


# -------------------------
# Features
# -------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["dt_hour"]
    df["hour"] = dt.dt.hour.astype(np.int16)
    df["dow"] = dt.dt.dayofweek.astype(np.int16)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    return df


def add_lag_roll(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    s = df["hourly_pow"].astype(np.float32)

    for l in [1, 24, 168]:
        df[f"lag_{l}"] = s.shift(l)

    df["roll_24"] = s.shift(1).rolling(24).mean()
    df["roll_168"] = s.shift(1).rolling(168).mean()
    return df


def make_model_frame(df_hour: pd.DataFrame) -> pd.DataFrame:
    df = df_hour.copy()
    df["dt_hour"] = pd.to_datetime(df["dt_hour"])
    df = add_time_features(df)
    df = add_lag_roll(df)
    df = df.dropna().reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in ["dt_hour", "hourly_pow"]]
    df[feat_cols] = df[feat_cols].astype(float)
    df["hourly_pow"] = df["hourly_pow"].astype(float)

    return df


# -------------------------
# Step 2) Train XGBoost
# -------------------------
def train_xgb(df_hour: pd.DataFrame):
    df = make_model_frame(df_hour)

    cutoff = df["dt_hour"].max() - pd.Timedelta(days=7)
    train = df[df["dt_hour"] <= cutoff]
    valid = df[df["dt_hour"] > cutoff]

    feat_cols = [c for c in df.columns if c not in ["dt_hour", "hourly_pow"]]
    X_train, y_train = train[feat_cols], train["hourly_pow"]
    X_valid, y_valid = valid[feat_cols], valid["hourly_pow"]

    model = XGBRegressor(
        n_estimators=1800,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.7,
        min_child_weight=5,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    return model, feat_cols


# -------------------------
# Recursive forecast for any horizon
#   - df_hist: 과거 관측(또는 예측으로 채운) hourly series
#   - future_idx: 예측할 dt_hour index
# -------------------------
def recursive_forecast(df_hist: pd.DataFrame, model: XGBRegressor, feat_cols: list[str], future_idx: pd.DatetimeIndex) -> pd.DataFrame:
    df_all = df_hist.copy()
    df_all["dt_hour"] = pd.to_datetime(df_all["dt_hour"])
    df_all = df_all.sort_values("dt_hour").reset_index(drop=True)

    # 미래 구간 append
    future_df = pd.DataFrame({"dt_hour": future_idx, "hourly_pow": np.nan})
    df_all = pd.concat([df_all, future_df], ignore_index=True)

    hist_len = len(df_hist)

    for i in range(hist_len, len(df_all)):
        temp = df_all.iloc[: i + 1].copy()
        temp = add_time_features(temp)
        temp = add_lag_roll(temp)

        row = temp.iloc[i]
        X = row[feat_cols].to_frame().T

        X = X.astype(float)
        if X.isna().any(axis=None):
            X = X.fillna(method="ffill", axis=0).fillna(0.0)

        pred = float(model.predict(X)[0])
        df_all.loc[i, "hourly_pow"] = max(pred, 0.0)

    out = df_all[df_all["dt_hour"].isin(future_idx)][["dt_hour", "hourly_pow"]].copy()
    return out


# -------------------------
# Save result
# -------------------------
def save_result(df_pred: pd.DataFrame, out_path: str, bill_rate=180.0, carbon_rate=0.424):
    df_out = df_pred.rename(columns={"dt_hour": "id"}).copy()
    df_out["id"] = pd.to_datetime(df_out["id"])

    agg_pow = float(df_out["hourly_pow"].sum())
    bill = agg_pow * float(bill_rate)
    carbon = agg_pow * float(carbon_rate)

    df_out["agg_pow"] = agg_pow
    df_out["may_bill"] = bill
    df_out["may_carbon"] = carbon

    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"rows={len(df_out)}, agg_pow={agg_pow:.6f}, bill={bill:.6f}, carbon={carbon:.6f}")


# -------------------------
# Metrics
# -------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# report
def print_team_report(april_true: pd.DataFrame, april_pred: pd.DataFrame, title_suffix: str = ""):
    """
     핵심: april_true에 존재하는 시간만 평가
      - merge(inner) => 실제 존재하는 시간대만 비교
    """
    df = april_true.merge(april_pred, on="dt_hour", how="inner", suffixes=("_true", "_pred")).sort_values("dt_hour")
    y_true = df["hourly_pow_true"].values
    y_pred = df["hourly_pow_pred"].values

    mae_v = mae(y_true, y_pred)
    rmse_v = rmse(y_true, y_pred)
    mape_v = mape(y_true, y_pred)
    smape_v = smape(y_true, y_pred)
    r2_v = r2_score(y_true, y_pred)

    total_true = float(np.sum(y_true))
    total_pred = float(np.sum(y_pred))
    diff_pct = float((total_pred - total_true) / (total_true if total_true != 0 else 1.0) * 100.0)

    print("\n팀 모델 성능 리포트 (XGBoost)" + (f" {title_suffix}" if title_suffix else ""))
    print("=======================================================")
    print("검증 기간: 20250401000000 ~ 20250430230000")
    print("-------------------------------------------------------")
    print("시간별 예측 오차")
    print(f"Aligned points : {len(df):,} (april_true에 존재하는 시간만)")
    print(f"MAE : {mae_v:.2f} kW")
    print(f"RMSE : {rmse_v:.2f} kW")
    print(f"MAPE : {mape_v:.4f} %")
    print(f"SMAPE : {smape_v:.4f} %")
    print(f"R^2 : {r2_v:.6f}")
    print("-------------------------------------------------------")
    print("총 사용량 정확도 (★가장 중요★)")
    print(f"실제 총 사용량 : {total_true:,.2f} kWh")
    print(f"예측 총 사용량 : {total_pred:,.2f} kWh")
    print(f"오차율 : {diff_pct:.5f} %")
    print("=======================================================\n")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, default="rtu_data_full.csv")
    parser.add_argument("--out_may", type=str, default="result.csv")
    parser.add_argument("--out_april", type=str, default="result_april.csv")
    parser.add_argument("--chunksize", type=int, default=3_000_000)
    parser.add_argument("--bill_rate", type=float, default=180.0)
    parser.add_argument("--carbon_rate", type=float, default=0.424)
    args = parser.parse_args()

    print("Step 1) Build hourly series split (train_hist excludes April, plus April_true)...")
    df_train_hist, df_april_true = build_hourly_series_split_april(
        args.raw_csv, args.chunksize,
        april_start="2025-04-01 00:00:00",
        april_end="2025-04-30 23:00:00",
    )

    print("TRAIN_HIST hourly:", df_train_hist.shape, df_train_hist["dt_hour"].min(), df_train_hist["dt_hour"].max())

    if df_april_true.empty:
        print("APRIL_TRUE hourly : EMPTY (원본 rtu에 4월 데이터가 없거나, 선택 범위가 없음)")
    else:
        print("APRIL_TRUE hourly :", df_april_true.shape, df_april_true["dt_hour"].min(), df_april_true["dt_hour"].max())
        print("APRIL_TRUE tail(5):")
        print(df_april_true.tail(5))

    print("Step 2) Train XGBoost (using TRAIN_HIST only, April excluded)...")
    model, feat_cols = train_xgb(df_train_hist)
    print("Model trained. Features:", len(feat_cols))

    # 2) 4월 예측 + 저장 + 성능 리포트
    print("Step 3) Forecast APRIL (2025-04-01 ~ 2025-04-30 23:00)...")
    april_idx = df_april_true["dt_hour"].sort_values().unique()
    df_april_pred = recursive_forecast(
        df_train_hist,
        model,
        feat_cols,
        pd.DatetimeIndex(april_idx)
    )
    print("Step 3-1) Save APRIL result_april.csv...")
    save_result(df_april_pred, args.out_april, bill_rate=args.bill_rate, carbon_rate=args.carbon_rate)

    # 리포트 출력 
    if not df_april_true.empty:
        print_team_report(df_april_true, df_april_pred)
    else:
        print("Skip report: APRIL_TRUE가 비어 있어서 성능평가를 할 수 없습니다.")

    # 1) 5월 예측 + 저장 (4월은 학습에서 배제, 대신 4월 예측값으로 브릿지)
    print("Step 4) Forecast MAY (672h: 2025-05-01 ~ 2025-05-28 23:00) using April-pred as bridge...")
    df_bridge = pd.concat([df_train_hist, df_april_pred], ignore_index=True).sort_values("dt_hour").reset_index(drop=True)

    may_idx = pd.date_range("2025-05-01 00:00:00", "2025-05-28 23:00:00", freq="h")
    df_may_pred = recursive_forecast(df_bridge, model, feat_cols, may_idx)

    print("Step 5) Save MAY result.csv...")
    save_result(df_may_pred, args.out_may, bill_rate=args.bill_rate, carbon_rate=args.carbon_rate)


if __name__ == "__main__":
    main()
