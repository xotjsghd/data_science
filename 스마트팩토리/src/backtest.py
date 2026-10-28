"""
Step 4~5) 월 단위 백테스트 — 모든 모델을 같은 기준으로 비교

  fold 2025-03 : ~2월 말까지 학습 → 3/1 00시 ~ 3/28 23시 (672시간) 예측
  fold 2025-04 : ~3월 말까지 학습 → 4/1 00시 ~ 4/28 23시 (672시간) 예측
  (대회와 같은 672시간 = 28일 구간, 월요일 시작 주 합산)

출력
  outputs/backtest_detail.csv   : fold × 모델별 전체 지표
  outputs/backtest_summary.csv  : 두 fold 평균 (score_total 내림차순)
  outputs/figures/backtest_daily.png
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import RAW_DIR, ROOT, load_module_hourly, load_module_operation, plant_series, target_index
from metrics import daily_sum, evaluate
from models import DESCRIPTIONS, MODELS

FOLDS = [("2025-03", "2025-03-01"), ("2025-04", "2025-04-01")]
OUT = ROOT / "outputs"


def run():
    wide = load_module_hourly()
    op = load_module_operation()
    plant = plant_series(wide)

    rows, preds = [], {}
    for fold, start in FOLDS:
        start = pd.Timestamp(start)
        target = target_index(start)
        hist, hist_op = wide[wide.index < start], op[op.index < start]
        truth = plant.reindex(target)
        assert truth.notna().all(), f"{fold}: 정답 구간에 빈 시간이 있음"
        preds[(fold, "actual")] = truth

        for name, fn in MODELS.items():
            t0 = time.perf_counter()
            p = fn(hist, hist_op, target)
            sec = time.perf_counter() - t0
            preds[(fold, name)] = p
            rows.append({"fold": fold, "model": name, "sec": sec, **evaluate(truth, p)})
            print(f"[{fold}] {name:<22} score={rows[-1]['score_total']:.4f} "
                  f"dSMAPE={rows[-1]['daily_smape_pct']:.3f}% agg={rows[-1]['agg_rel_err_pct']:+.3f}% ({sec:.1f}s)",
                  flush=True)

        # 참고: 사후적으로 정답 월평균을 알았을 때의 상수 예측
        p = pd.Series(truth.mean(), index=target)
        rows.append({"fold": fold, "model": "oracle_flat", "sec": np.nan, **evaluate(truth, p)})

    detail = pd.DataFrame(rows)
    detail.insert(2, "description", detail["model"].map(DESCRIPTIONS))
    OUT.mkdir(exist_ok=True)
    detail.to_csv(OUT / "backtest_detail.csv", index=False, encoding="utf-8-sig")

    detail["abs_agg_err_pct"] = detail["agg_rel_err_pct"].abs()
    cols = ["score_total", "daily_score", "weekly_score", "bill_score",
            "daily_smape_pct", "weekly_smape_pct", "daily_nmae_pct", "abs_agg_err_pct",
            "hourly_mae", "hourly_smape_pct", "sec"]
    summary = detail.groupby("model")[cols].mean()
    summary.insert(0, "description", summary.index.map(DESCRIPTIONS))
    per_fold = detail.pivot(index="model", columns="fold", values="agg_rel_err_pct")
    per_fold.columns = [f"agg_err_{c}_pct" for c in per_fold.columns]
    summary = summary.join(per_fold).sort_values("score_total", ascending=False)
    summary.to_csv(OUT / "backtest_summary.csv", encoding="utf-8-sig")

    save_figure(preds)
    return summary, wide, op, plant


def submitted_check(plant):
    """
    당시 실제로 만든 4월 예측 파일(../xgboost_result_april.csv)을 같은 지표·같은 672시간으로 채점.
    재현 모델(orig_xgb_recursive)이 당시 결과와 같은 수준인지, 개선본과 얼마나 차이 나는지 확인한다.
    """
    path = RAW_DIR / "xgboost_result_april.csv"
    if not path.exists():
        return None
    target = target_index("2025-04-01")
    sub = pd.read_csv(path, parse_dates=["id"]).set_index("id")["hourly_pow"].reindex(target)
    truth = plant.reindex(target)
    detail = pd.read_csv(OUT / "backtest_detail.csv")
    rows = [{"model": "당시 제출 파이프라인 4월 예측 (xgboost_result_april.csv)", **evaluate(truth, sub)}]
    for m in ["orig_xgb_recursive", "base_mean_all"]:
        r = detail[(detail["fold"] == "2025-04") & (detail["model"] == m)].iloc[0]
        rows.append({"model": m, **{k: r[k] for k in evaluate(truth, sub)}})
    out = pd.DataFrame(rows).set_index("model")
    out.to_csv(OUT / "submitted_check.csv", encoding="utf-8-sig")
    return out


ROLLING_MODELS = ["orig_xgb_recursive", "base_mean_all", "base_mean_4w", "base_dowhour_4w",
                  "lgbm_direct", "lgbm_direct_module"]


def rolling_check(wide, op, plant):
    """
    보조 검증: 두 fold만으로는 잡음에 순위가 흔들릴 수 있으므로
    2/1 ~ 4/1 사이 매주 토요일을 예측 시작점으로 672시간 예측을 9번 반복해 순위 안정성을 확인.
    """
    rows = []
    for start in pd.date_range("2025-02-01", "2025-04-01", freq="7D"):
        target = target_index(start)
        hist, hist_op = wide[wide.index < start], op[op.index < start]
        truth = plant.reindex(target)
        for name in ROLLING_MODELS:
            r = evaluate(truth, MODELS[name](hist, hist_op, target))
            rows.append({"origin": start.date(), "model": name, **r})
    df = pd.DataFrame(rows)
    df["abs_agg_err_pct"] = df["agg_rel_err_pct"].abs()
    metrics = ["daily_smape_pct", "weekly_smape_pct", "abs_agg_err_pct", "score_total"]
    out = df.groupby("model")[metrics].mean()
    base = df[df["model"] == "orig_xgb_recursive"].set_index("origin")
    for m in out.index:
        cur = df[df["model"] == m].set_index("origin")
        out.loc[m, "daily_win_vs_orig"] = (cur["daily_smape_pct"] < base["daily_smape_pct"]).sum()
        out.loc[m, "weekly_win_vs_orig"] = (cur["weekly_smape_pct"] < base["weekly_smape_pct"]).sum()
    out["n_origins"] = df["origin"].nunique()
    out = out.sort_values("daily_smape_pct")
    out.to_csv(OUT / "rolling_check.csv", encoding="utf-8-sig")
    return out


def save_figure(preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, len(FOLDS), figsize=(14, 4.2), sharey=True)
    show = [("actual", "실측", "k", 2.2), ("orig_xgb_recursive", "당시 방식 재현(XGB 재귀)", "tab:red", 1.4),
            ("base_dowhour_4w", "최근 4주 요일×시간 평균", "tab:orange", 1.2),
            ("lgbm_direct", "LightGBM 직접(lag≥672)", "tab:green", 1.2),
            ("base_mean_all", "전체 기간 평균(최종)", "tab:blue", 1.6)]
    for ax, (fold, _) in zip(axes, FOLDS):
        for key, label, color, lw in show:
            d = daily_sum(preds[(fold, key)])
            ax.plot(d.index, d.values, label=label, color=color, lw=lw)
        ax.set_title(f"{fold} 백테스트 — 일별 합산 (kWh)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("일 사용량 (kWh)")
    lo, hi = axes[0].get_ylim()
    fig.suptitle(f"세로축 전체 폭 {hi - lo:,.0f} kWh = 일 사용량의 약 {(hi - lo) / np.mean([lo, hi]):.2%} "
                 "— 선 사이 차이는 모두 측정 잡음 수준", fontsize=10, color="0.35")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "backtest_daily.png", dpi=130)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    s, wide, op, plant = run()
    show = ["score_total", "daily_smape_pct", "weekly_smape_pct", "abs_agg_err_pct", "bill_score", "hourly_mae", "sec"]
    print(s[show].round(4).to_string())
    print("\n[보조] 주 단위 롤링 시작점 9개 평균")
    print(rolling_check(wide, op, plant).round(4).to_string())
    chk = submitted_check(plant)
    if chk is not None:
        print("\n[당시 제출 파일 채점] 4/1~4/28 (672h)")
        print(chk[["daily_smape_pct", "weekly_smape_pct", "agg_rel_err_pct", "hourly_mae", "score_total"]]
              .round(4).to_string())
