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

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_FILE = "dataset_06_equipment_failure_warning.csv"

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 60)
print("EQUIPMENT FAILURE WARNING - LOGISTIC REGRESSION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())


# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nTarget percentages:")
print(df["target"].value_counts(normalize=True).sort_index() * 100)


# ============================================================
# 3. FEATURE INFORMATION
# ============================================================

features = [
    "temperature",
    "vibration",
    "pressure",
    "runtime_hours",
    "maintenance_gap_days",
    "power_draw"
]

target = "target"

X = df[features].copy()
y = df[target].astype(int)


# ============================================================
# 4. FEATURE RANGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("FEATURE RANGES")
print("=" * 60)

for feature in features:
    print(
        f"{feature:25s} "
        f"Min = {X[feature].min():.2f}   "
        f"Max = {X[feature].max():.2f}"
    )


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

fig = plt.figure(figsize=(8, 6))

target_counts = y.value_counts().sort_index()

bars = plt.bar(
    ["Non-Failure (0)", "Failure Risk (1)"],
    target_counts.values
)

plt.title("Equipment Failure Risk - Target Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Records")

for bar, value in zip(bars, target_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 10,
        str(value),
        ha="center",
        fontsize=11
    )

plt.tight_layout()
fig.savefig(
    "01_target_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.close(fig)

print("\nSaved: 01_target_distribution.png")


# ============================================================
# 6. FEATURE DISTRIBUTIONS
# ============================================================

fig = plt.figure(figsize=(14, 9))

for i, feature in enumerate(features, start=1):

    ax = fig.add_subplot(2, 3, i)

    ax.hist(
        X[feature],
        bins=25,
        edgecolor="black"
    )

    ax.set_title(feature.replace("_", " ").title())
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Feature Distributions",
    fontsize=16
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(
    "02_feature_distributions.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.close(fig)

print("Saved: 02_feature_distributions.png")


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

correlation_data = df[features + [target]]
correlation_matrix = correlation_data.corr()

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)

image = ax.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

ax.set_title("Correlation Matrix", fontsize=16)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))

ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    correlation_matrix.columns
)

# Add correlation values
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):

        value = correlation_matrix.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

fig.colorbar(image, ax=ax)

fig.tight_layout()

fig.savefig(
    "03_correlation_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.close(fig)

print("Saved: 03_correlation_matrix.png")


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 9. BASELINE MODEL
# ============================================================

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nBaseline Accuracy:")
print(f"{baseline_accuracy:.6f}")


# ============================================================
# 10. LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
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
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# ============================================================
# 12. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

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

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-Score  : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")

print("\nPercentage Results:")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Failure",
            "Failure Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

fig = plt.figure(figsize=(8, 7))

ax = fig.add_subplot(111)

matrix_image = ax.imshow(
    cm,
    interpolation="nearest",
    aspect="auto"
)

ax.set_title(
    "Confusion Matrix - Equipment Failure Prediction",
    fontsize=15
)

ax.set_xlabel(
    "Predicted Label",
    fontsize=12
)

ax.set_ylabel(
    "Actual Label",
    fontsize=12
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Non-Failure",
    "Failure Risk"
])

ax.set_yticklabels([
    "Non-Failure",
    "Failure Risk"
])

# Add values inside matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold"
        )

fig.colorbar(
    matrix_image,
    ax=ax
)

fig.tight_layout()

fig.savefig(
    "04_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("\nSaved: 04_confusion_matrix.png")


# ============================================================
# 16. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

fig = plt.figure(figsize=(8, 7))

ax = fig.add_subplot(111)

ax.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)

ax.set_title(
    "ROC Curve - Equipment Failure Prediction",
    fontsize=15
)

ax.set_xlabel(
    "False Positive Rate",
    fontsize=12
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=12
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

ax.grid(alpha=0.3)
ax.legend(loc="lower right")

fig.tight_layout()

fig.savefig(
    "05_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("Saved: 05_roc_curve.png")


# ============================================================
# 17. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

logistic_model = model.named_steps[
    "logistic_regression"
]

coefficients = logistic_model.coef_[0]

odds_ratios = np.exp(coefficients)

coefficient_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients,
    "Odds_Ratio": odds_ratios
})

coefficient_df = coefficient_df.sort_values(
    by="Coefficient",
    ascending=True
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

print(
    coefficient_df.to_string(
        index=False
    )
)


# ============================================================
# 18. COEFFICIENT PLOT
# ============================================================

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111)

ax.barh(
    coefficient_df["Feature"],
    coefficient_df["Coefficient"],
    edgecolor="black"
)

ax.axvline(
    x=0,
    linewidth=1.5
)

ax.set_title(
    "Logistic Regression Feature Coefficients",
    fontsize=15
)

ax.set_xlabel(
    "Coefficient",
    fontsize=12
)

ax.set_ylabel(
    "Feature",
    fontsize=12
)

ax.grid(
    axis="x",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    "06_coefficients.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("\nSaved: 06_coefficients.png")


# ============================================================
# 19. SAVE COEFFICIENTS
# ============================================================

coefficient_df.to_csv(
    "logistic_coefficients.csv",
    index=False
)

print("Saved: logistic_coefficients.csv")


# ============================================================
# 20. SAVE PREDICTIONS
# ============================================================

predictions_df = X_test.copy()

predictions_df["actual_target"] = y_test.values

predictions_df["predicted_target"] = y_pred

predictions_df["failure_probability"] = y_probability

predictions_df.to_csv(
    "equipment_failure_predictions.csv",
    index=False
)

print("Saved: equipment_failure_predictions.csv")


# ============================================================
# 21. SAVE CLEANED DATASET
# ============================================================

cleaned_df = df[
    features + [target]
].copy()

cleaned_df.to_csv(
    "equipment_failure_cleaned.csv",
    index=False
)

print("Saved: equipment_failure_cleaned.csv")


# ============================================================
# 22. SAVE MODEL RESULTS
# ============================================================

with open(
    "model_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EQUIPMENT FAILURE WARNING - LOGISTIC REGRESSION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Baseline Accuracy: {baseline_accuracy:.6f}\n\n"
    )

    file.write(
        "Logistic Regression Performance\n"
    )

    file.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    file.write(
        f"Precision: {precision:.6f}\n"
    )

    file.write(
        f"Recall: {recall:.6f}\n"
    )

    file.write(
        f"F1-Score: {f1:.6f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.6f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm) + "\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Non-Failure",
                "Failure Risk"
            ],
            zero_division=0
        )
    )

print("Saved: model_results.txt")


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated visualization files:")

print("1. 01_target_distribution.png")
print("2. 02_feature_distributions.png")
print("3. 03_correlation_matrix.png")
print("4. 04_confusion_matrix.png")
print("5. 05_roc_curve.png")
print("6. 06_coefficients.png")

print("\nGenerated result files:")

print("7. equipment_failure_cleaned.csv")
print("8. equipment_failure_predictions.csv")
print("9. logistic_coefficients.csv")
print("10. model_results.txt")

print("\nAll files have been generated successfully.")