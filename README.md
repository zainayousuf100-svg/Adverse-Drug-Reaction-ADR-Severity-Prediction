# Adverse Drug Reaction (ADR) Severity Prediction

**Author:** Syeda Zaina Yousuf
**Tools:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Overview
This project predicts whether a reported Adverse Drug Reaction (ADR) is likely
to be classified as **Serious** vs **Non-Serious**, based on patient and
prescription-level features — drug class, age, dosage, treatment duration,
comorbidities, polypharmacy, and prior ADR history.

The project draws directly on my pharmacovigilance background: during my
healthcare internship I documented ADR case reports in compliance with
**ICH-GCP and ICSR (Individual Case Safety Report)** standards. This project
translates that domain knowledge into a data science workflow — using the
same categories of variables that real pharmacovigilance case reports
capture.

## Why this matters
Regulatory bodies (like the FDA and CDSCO) and pharma companies process
thousands of ADR reports. Being able to flag which incoming reports are
likely *serious* — meaning they may involve hospitalization, disability, or
life-threatening outcomes — helps safety teams **triage and prioritize
review faster**, which is exactly the kind of problem data science is
starting to solve in drug safety monitoring.

## Methodology
1. **Data:** A synthetic dataset (1,200 simulated case reports) built to
   reflect realistic pharmacovigilance risk patterns — e.g., anticoagulants
   and chemotherapy agents carry a higher baseline risk of serious ADRs,
   consistent with real-world reporting trends. *(See note below on using
   real data.)*
2. **EDA:** Visualized ADR seriousness rates by drug class, age distribution,
   polypharmacy/comorbidity interaction, and feature correlations.
3. **Modeling:** Trained and compared two classifiers:
   - Logistic Regression (interpretable baseline)
   - Random Forest (captures non-linear feature interactions)
4. **Evaluation:** Accuracy, ROC-AUC, precision/recall, confusion matrix, and
   feature importance ranking.

## Results
- Random Forest achieved ROC-AUC ≈ 0.68, outperforming the logistic baseline.
- Drug class, comorbidity count, and polypharmacy count emerged as the
  strongest predictors of ADR seriousness — consistent with established
  pharmacovigilance risk factors (drug interactions and multi-morbidity are
  well-documented contributors to serious adverse events).

## Files
- `adr_severity_prediction.py` — full pipeline: data generation, EDA, modeling, evaluation
- `adr_dataset.csv` — the generated dataset
- `eda_overview.png` — exploratory data analysis visuals
- `model_evaluation.png` — confusion matrix, ROC curves, feature importance

