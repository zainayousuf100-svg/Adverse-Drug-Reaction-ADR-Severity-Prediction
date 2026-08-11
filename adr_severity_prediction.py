"""
Adverse Drug Reaction (ADR) Severity Prediction
=================================================
Author: Syeda Zaina Yousuf

Goal: Predict whether a reported Adverse Drug Reaction (ADR) is likely to be
"Serious" vs "Non-Serious" based on patient and prescription-level features,
using patterns consistent with real-world pharmacovigilance data
(as documented under ICH-GCP / ICSR reporting standards).

NOTE ON DATA: This script generates a realistic SYNTHETIC dataset that
mirrors the structure of real ADR/ICSR reports (patient demographics, drug
class, dosage, duration, comorbidities, polypharmacy). This is a common and
accepted practice for a portfolio project when real patient data is
confidential. For a production version, this pipeline is designed to be
dropped directly onto a real dataset such as the FDA FAERS (FDA Adverse
Event Reporting System) public dataset -- see README for details.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)

sns.set_style("whitegrid")
RNG = np.random.default_rng(42)
N = 1200  # number of simulated ADR case reports

# ---------------------------------------------------------------------------
# 1. GENERATE A REALISTIC SYNTHETIC ADR DATASET
# ---------------------------------------------------------------------------
drug_classes = ["Antibiotics", "NSAIDs", "Antihypertensives", "Antidiabetics",
                 "Anticoagulants", "Chemotherapy", "CNS/Psychiatric", "Antihistamines"]

# Base seriousness risk per drug class (reflects real pharmacovigilance patterns:
# anticoagulants & chemotherapy agents carry higher serious-ADR risk)
class_risk = {
    "Antibiotics": 0.20, "NSAIDs": 0.25, "Antihypertensives": 0.22,
    "Antidiabetics": 0.28, "Anticoagulants": 0.55, "Chemotherapy": 0.60,
    "CNS/Psychiatric": 0.35, "Antihistamines": 0.10
}

age = RNG.integers(1, 90, N)
gender = RNG.choice(["Male", "Female"], N)
drug_class = RNG.choice(drug_classes, N)
dosage_mg = np.round(RNG.uniform(5, 1000, N), 1)
duration_days = RNG.integers(1, 180, N)
comorbidity_count = RNG.poisson(1.2, N)
polypharmacy_count = RNG.poisson(2.5, N)  # number of concurrent medications
prior_adr_history = RNG.choice([0, 1], N, p=[0.85, 0.15])

# Build seriousness probability from clinically plausible risk factors
base_risk = np.array([class_risk[d] for d in drug_class])
age_factor = np.where(age > 65, 0.15, np.where(age < 12, 0.10, 0.0))
comorbidity_factor = comorbidity_count * 0.04
polypharmacy_factor = polypharmacy_count * 0.03
duration_factor = np.where(duration_days > 90, 0.08, 0.0)
history_factor = prior_adr_history * 0.12

risk_score = (base_risk + age_factor + comorbidity_factor +
              polypharmacy_factor + duration_factor + history_factor)
risk_score = np.clip(risk_score + RNG.normal(0, 0.08, N), 0, 1)

serious = (RNG.uniform(0, 1, N) < risk_score).astype(int)

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "drug_class": drug_class,
    "dosage_mg": dosage_mg,
    "duration_days": duration_days,
    "comorbidity_count": comorbidity_count,
    "polypharmacy_count": polypharmacy_count,
    "prior_adr_history": prior_adr_history,
    "serious_adr": serious  # target: 1 = Serious, 0 = Non-Serious
})

df.to_csv("/home/claude/adr_project/adr_dataset.csv", index=False)
print("Dataset shape:", df.shape)
print(df["serious_adr"].value_counts(normalize=True).round(2))

# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.barplot(data=df, x="drug_class", y="serious_adr", ax=axes[0, 0],
            estimator=np.mean, errorbar=None, hue="drug_class", palette="viridis", legend=False)
axes[0, 0].set_title("Serious ADR Rate by Drug Class")
axes[0, 0].set_ylabel("Proportion Serious")
axes[0, 0].tick_params(axis='x', rotation=40)

sns.boxplot(data=df, x="serious_adr", y="age", ax=axes[0, 1], hue="serious_adr",
            palette="Set2", legend=False)
axes[0, 1].set_title("Age Distribution: Serious vs Non-Serious ADR")
axes[0, 1].set_xticklabels(["Non-Serious", "Serious"])

sns.scatterplot(data=df, x="polypharmacy_count", y="comorbidity_count",
                 hue="serious_adr", palette="coolwarm", alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title("Polypharmacy vs Comorbidity (colored by seriousness)")

corr_cols = ["age", "dosage_mg", "duration_days", "comorbidity_count",
             "polypharmacy_count", "prior_adr_history", "serious_adr"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("/home/claude/adr_project/eda_overview.png", dpi=150)
plt.close()
print("Saved eda_overview.png")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING + MODELING
# ---------------------------------------------------------------------------
data = df.copy()
le_gender = LabelEncoder()
le_drug = LabelEncoder()
data["gender_enc"] = le_gender.fit_transform(data["gender"])
data["drug_class_enc"] = le_drug.fit_transform(data["drug_class"])

features = ["age", "gender_enc", "drug_class_enc", "dosage_mg", "duration_days",
            "comorbidity_count", "polypharmacy_count", "prior_adr_history"]
X = data[features]
y = data["serious_adr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression (interpretable baseline)
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest (captures non-linear interactions)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42,
                             class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_lr), 3))
print(classification_report(y_test, y_pred_lr, target_names=["Non-Serious", "Serious"]))

print("\n=== Random Forest ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_rf), 3))
print(classification_report(y_test, y_pred_rf, target_names=["Non-Serious", "Serious"]))

# ---------------------------------------------------------------------------
# 4. MODEL EVALUATION VISUALS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Confusion matrix (Random Forest)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Serious", "Serious"],
            yticklabels=["Non-Serious", "Serious"], ax=axes[0])
axes[0].set_title("Random Forest: Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
axes[1].plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC={roc_auc_score(y_test, y_prob_lr):.2f})")
axes[1].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, y_prob_rf):.2f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curve Comparison")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
importances.plot(kind="barh", ax=axes[2], color="teal")
axes[2].set_title("Feature Importance (Random Forest)")

plt.tight_layout()
plt.savefig("/home/claude/adr_project/model_evaluation.png", dpi=150)
plt.close()
print("Saved model_evaluation.png")

print("\nProject run complete. Outputs: adr_dataset.csv, eda_overview.png, model_evaluation.png")
