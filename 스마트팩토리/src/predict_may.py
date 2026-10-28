"""
Step 6) 최종 모델로 4월 실측까지 포함해 재학습 → 5월 672시간 제출 파일 생성

  - 기본 모델: base_mean_all (백테스트 2개 fold + 롤링 9개 구간에서 가장 낮은 오차)
  - 학습 데이터: 2024-12-01 00시 ~ 2025-04-29 23시 (4/30 00시는 샘플 1개뿐이라 제외)
  - 출력: outputs/submission_may.csv  (id, hourly_pow[kWh], agg_pow[kWh], may_bill[원], may_carbon[kgCO2])

사용 예)
  python src/predict_may.py                      # 기본(base_mean_all)
  python src/predict_may.py --model lgbm_direct  # 다른 모델로 생성
"""
import argparse

import pandas as pd

from data import RAW_DIR, ROOT, load_module_hourly, load_module_operation, target_index
from metrics import make_submission
from models import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base_mean_all", choices=list(MODELS))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "submission_may.csv"))
    args = ap.parse_args()

    wide, op = load_module_hourly(), load_module_operation()
    may = target_index("2025-05-01")
    print(f"학습 구간: {wide.index.min()} ~ {wide.index.max()} ({len(wide):,}시간, 설비 {wide.shape[1]}개)")

    pred = MODELS[args.model](wide, op, may)
    assert len(pred) == 672 and pred.notna().all()
    sub = make_submission(pred)
    sub.to_csv(args.out, index=False)

    agg = sub["agg_pow"].iloc[0]
    print(f"모델: {args.model}")
    print(f"저장: {args.out}  ({len(sub)}행, {sub['id'].iloc[0]} ~ {sub['id'].iloc[-1]})")
    print(f"hourly_pow 평균 {sub['hourly_pow'].mean():,.1f} kWh/h")
    print(f"agg_pow   = {agg:,.0f} kWh")
    print(f"may_bill  = {sub['may_bill'].iloc[0]:,.0f} 원")
    print(f"may_carbon= {sub['may_carbon'].iloc[0]:,.0f} kgCO2")

    orig_path = RAW_DIR / "xgboost_result.csv"
    if orig_path.exists():
        o = pd.read_csv(orig_path)
        o_agg = o["agg_pow"].iloc[0]
        print("\n[공모전 제출본(xgboost_result.csv)과 비교]")
        print(f"agg_pow   : 제출본 {o_agg:,.0f} → 개선본 {agg:,.0f} kWh ({(agg - o_agg) / o_agg:+.3%})")
        print(f"may_bill  : 제출본 {o['may_bill'].iloc[0]:,.0f} → 개선본 {sub['may_bill'].iloc[0]:,.0f} 원")
        print(f"hourly_pow: 제출본 {o['hourly_pow'].min():,.0f}~{o['hourly_pow'].max():,.0f} "
              f"(평균 {o['hourly_pow'].mean():,.1f}) → 개선본 {sub['hourly_pow'].mean():,.1f} 고정")


if __name__ == "__main__":
    main()
