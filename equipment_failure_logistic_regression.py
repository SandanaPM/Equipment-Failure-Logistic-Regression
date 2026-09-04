import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# 1. Load dataset
df = pd.read_csv("dataset_06_equipment_failure_warning.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

# 2. Data quality checks
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

# 3. Feature range checks
print("\nMinimum values:")
print(df.min(numeric_only=True))

print("\nMaximum values:")
print(df.max(numeric_only=True))

# 4. Exploratory visualizations
plt.figure(figsize=(6, 4))
df["target"].value_counts().sort_index().plot(kind="bar")
plt.title("Equipment Failure Target Distribution")
plt.xlabel("Target (0 = No Failure, 1 = Failure)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

df[
    [
        "temperature",
        "vibration",
        "pressure",
        "runtime_hours",
        "maintenance_gap_days",
        "power_draw"
    ]
].hist(figsize=(12, 8), bins=25)

plt.suptitle("Input Feature Distributions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 7))

correlation = df.corr(numeric_only=True)

plt.imshow(correlation, aspect="auto")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.colorbar()
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 5. Define features and target
features = [
    "temperature",
    "vibration",
    "pressure",
    "runtime_hours",
    "maintenance_gap_days",
    "power_draw"
]

X = df[features]
y = df["target"].astype(int)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:")
print(X_train.shape)

print("\nTesting shape:")
print(X_test.shape)

# 7. Simple baseline
baseline = DummyClassifier(
    strategy="most_frequent",
    random_state=42
)

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

print("\nBaseline Accuracy:")
print(baseline_accuracy)

# 8. Logistic Regression pipeline
model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

# 9. Train model
model.fit(
    X_train,
    y_train
)

print("\nLogistic Regression model trained successfully.")

# 10. Predictions
y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred,
    "Failure_Probability": y_probability
})

print("\nActual vs Predicted:")
print(results.head(10))

# 11. Evaluation
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nModel Evaluation")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

# 12. Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Failure",
            "Failure"
        ],
        zero_division=0
    )
)

# 13. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.xticks(
    [0, 1],
    ["No Failure", "Failure"]
)

plt.yticks(
    [0, 1],
    ["No Failure", "Failure"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.show()

# 14. ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# 15. Coefficient interpretation
lr = model.named_steps[
    "logistic_regression"
]

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": lr.coef_[0],
    "Odds_Ratio": np.exp(lr.coef_[0])
})

coefficients = coefficients.sort_values(
    "Coefficient",
    ascending=False
)

print("\nLogistic Regression Coefficients:")
print(coefficients)

# 16. Save outputs
df.to_csv(
    "equipment_failure_cleaned.csv",
    index=False
)

results.to_csv(
    "equipment_failure_predictions.csv",
    index=False
)

coefficients.to_csv(
    "logistic_coefficients.csv",
    index=False
)

print("\nOutput files created:")

print("1. equipment_failure_cleaned.csv")
print("2. equipment_failure_predictions.csv")
print("3. logistic_coefficients.csv")

print(
    "\nEquipment Failure Warning Logistic Regression "
    "project completed successfully."
)
plt.savefig("01_target_distribution.png")
plt.savefig("02_feature_distributions.png")
plt.savefig("03_correlation_matrix.png")
plt.savefig("04_confusion_matrix.png")
plt.savefig("05_roc_curve.png")
plt.savefig("06_coefficients.png")
