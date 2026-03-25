# WhyLab Advanced Analysis Report

**Generated At**: {generated_at}
**Scenario**: {scenario}
**Selected Model**: {model_type}

---

## 1. AutoML Competition Result
본 실험에서는 데이터의 특성에 가장 적합한 인과추론 모델을 찾기 위해 **AutoML Competition**을 수행했습니다.

| Rank | Model Type | RMSE (CATE) | Win? |
|------|------------|-------------|------|
| 1st | **{model_type}** | **{best_score:.4f}** | 🏆 |
| 2nd | {comp_model_1} | {comp_score_1:.4f} | |

> **Analysis**: 선정된 모델은 경쟁 모델 대비 약 **{improvement_ratio:.1f}%** 더 낮은 예측 오차를 보였습니다. 이는 데이터의 {nonlinear_reason} 특성을 더 잘 포착했기 때문입니다.

---

## 2. Robustness Check (Sensitivity Analysis)
도출된 인과 효과가 단순한 우연이나 편향에 의한 것이 아님을 입증합니다.

### 2.1. Placebo Treatment Test
- **Method**: 처치 변수를 무작위로 섞어 효과가 사라지는지 테스트.
- **Result**: Estimated Effect = {placebo_effect:.5f} (p-value = {placebo_p:.3f})
- **Status**: {placebo_status_badge}

### 2.2. Random Common Cause Test
- **Method**: 무작위 교란 변수를 추가해도 결과가 유지되는지 테스트.
- **Result**: Stability Score = {rcc_stability:.2f}
- **Status**: {rcc_status_badge}

> **Verdict**: 본 실험의 결과는 **{final_verdict}**합니다.

---

## 3. Deep Dive: Feature Importance (SHAP)
어떤 변수가 인과 효과(CATE)에 가장 큰 영향을 미쳤나요?

1.  **{top_feature_1}**: 높을수록 처치 효과가 {feat_1_direction}.
2.  **{top_feature_2}**: 특정 구간에서 이질성이 큼.

---

*Powered by WhyLab 2.0*
