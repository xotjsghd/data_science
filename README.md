# Data Science Projects

데이터 분석·머신러닝 대회와 수업 프로젝트를 정리한 저장소이다.
각 폴더에 독립된 README가 있고, 대회 데이터는 배포 정책에 따라 포함하지 않았다.

| 프로젝트 | 대회 / 과제 | 문제 유형 | 한 줄 요약 |
|---|---|---|---|
| [🔥 Wildfire Survival Prediction](Wildfire%20Survival%20Prediction/) | WiDS Global Datathon 2026 (팀) | 생존분석 (time-to-event) | 산불 발생 후 5시간 관측으로 12~72시간 안에 대피구역에 도달할 확률 예측 |
| [🏭 스마트팩토리 전력소비 예측](%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8C%A9%ED%86%A0%EB%A6%AC/) | 제4회 BDA 채용공모전 (팀) + 개인 개선판 | 시계열 예측 | 공장 설비 13개의 5초 단위 센서 데이터로 5월 한 달 시간별 전력·요금·탄소배출 예측 |

---

## 🔥 Wildfire Survival Prediction
**WiDS Global Datathon 2026** · 2026 봄학기 데이터과학 프로젝트 22조 (3인)

- 산불이 대피구역에 "도달하느냐"가 아니라 **"언제 도달하느냐"** 를 다루는 생존분석 문제로 정의
- 221건 중 68.8%가 중도절단(right-censored)된 데이터에서 Stratified 5-Fold OOF 검증
- 생존함수에서 시점별 도달 확률 P(T≤H) = 1 − S(H)를 뽑고, 시점 간 단조성 보정
- **Lasso-Cox**(해석 가능한 기준 모델) + **Random Survival Forest**(비선형) → 0.8 : 0.2 가중 앙상블

| 모델 | OOF Hybrid Score | Public Score |
|---|---:|---:|
| Lasso-Cox | 0.9510 | 0.9358 |
| Random Survival Forest | 0.9727 | 0.9623 |
| **Ensemble** | **0.9722** | **0.9632** |

Kaggle 리더보드 최종 순위는 1,754팀 중 788위(Private, 상위 45%)였다. Public 기준 855위에서 67계단 올랐다.

`Python` `Survival Analysis` `Lasso-Cox` `Random Survival Forest` → [자세히 보기](Wildfire%20Survival%20Prediction/)

---

## 🏭 스마트팩토리 전력소비 예측
**제4회 BDA 채용공모전** · 팀 참가 (2025.12 ~ 2026.01) → 개인 재검증·개선 (2026.10)

- 4.9GB, 3,370만 행 원본을 한 번만 스트리밍 집계해 49초 만에 설비×시간 데이터로 정리
- 데이터 진단 결과 공장 전력의 시간별 변동이 **측정 잡음 크기와 같아서** 시간대·요일·공휴일 패턴이 없음을 확인
- "4월 결측"의 원인이 **엑셀 최대 행 수에서 잘린 `test.csv`** 였음을 추적
- 대회 평가 지표(일·주 합산, 요금·탄소 Bell-curve)를 구현하고 월 단위 백테스트 + 롤링 9회로 모델 11종 비교
- 결론: 당시 채택한 XGBoost 재귀 예측보다 **학습 기간 전체 평균**이 일관되게 정확 (롤링 9회 중 일별 기준 9회)
  — 절대적인 차이가 작다는 점까지 함께 기록

`Python` `pandas` `pyarrow` `XGBoost` `LightGBM` `Time Series` `Backtesting` → [자세히 보기](%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8C%A9%ED%86%A0%EB%A6%AC/)

---

## 📁 구조
```
data_science/
├─ Wildfire Survival Prediction/   # 노트북(EDA·Lasso-Cox·RSF·앙상블), OOF 예측, 최종 보고서·발표자료
└─ 스마트팩토리/                    # src/ 파이프라인, 데이터 진단 리포트, 백테스트 결과, legacy/ 당시 코드
```
