import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

train = pd.read_csv("train.csv")
y_time  = train["time_to_hit_hours"].values
y_event = train["event"].values

# ── 각 노트북에서 저장한 OOF 배열을 여기에 붙여넣기 ──────────
# RSF 노트북:
#   best_res = run_rsf_cv(...)
#   np.save("rsf_oof_prob.npy", best_res["oof_prob"])
#   np.save("rsf_oof_risk.npy", best_res["oof_risk"])
#
# Cox 노트북:
#   np.save("cox_oof_prob.npy", best_res["oof_prob"])
#   np.save("cox_oof_risk.npy", best_res["oof_risk"])
# ────────────────────────────────────────────────────────────

rsf_oof_prob = np.load("rsf_oof_prob.npy")   # shape (221, 4)
rsf_oof_risk = np.load("rsf_oof_risk.npy")   # shape (221,)
cox_oof_prob = np.load("cox_oof_prob.npy")   # shape (221, 4)
cox_oof_risk = np.load("cox_oof_risk.npy")   # shape (221,)

W = 0.8   # RSF 가중치

# 앙상블 OOF
ens_prob = W * rsf_oof_prob + (1 - W) * cox_oof_prob
ens_prob = np.maximum.accumulate(ens_prob, axis=1)   # 단조성 보정
ens_risk = W * rsf_oof_risk + (1 - W) * cox_oof_risk

# ── Brier Score ───────────────────────────────────────────
def brier(times, events, pred, H):
    times, events, pred = map(np.asarray, [times, events, pred])
    mask   = (events == 1) | ((events == 0) & (times >= H))
    y_true = ((events == 1) & (times <= H)).astype(float)
    if mask.sum() == 0:
        return np.nan, 0
    return float(np.mean((y_true[mask] - pred[mask]) ** 2)), int(mask.sum())

b24, n24 = brier(y_time, y_event, ens_prob[:, 1], 24)
b48, n48 = brier(y_time, y_event, ens_prob[:, 2], 48)
b72, n72 = brier(y_time, y_event, ens_prob[:, 3], 72)
wb       = 0.3 * b24 + 0.4 * b48 + 0.3 * b72

# ── C-index ───────────────────────────────────────────────
ci = concordance_index(y_time, -ens_risk, y_event)

# ── Hybrid Score ──────────────────────────────────────────
hs = 0.3 * ci + 0.7 * (1 - wb)

print("=" * 45)
print(f"  Ensemble OOF (RSF {W:.0%} + Cox {1-W:.0%})")
print("=" * 45)
print(f"  C-index        : {ci:.6f}")
print(f"  Brier@24h      : {b24:.6f}  (n={n24})")
print(f"  Brier@48h      : {b48:.6f}  (n={n48})")
print(f"  Brier@72h      : {b72:.6f}  (n={n72})")
print(f"  Weighted Brier : {wb:.6f}")
print(f"  Hybrid Score   : {hs:.6f}")
print("=" * 45)
