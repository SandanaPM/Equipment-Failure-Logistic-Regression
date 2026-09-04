# Equipment Failure Warning using Logistic Regression

## Project Overview

This project uses Machine Learning to predict equipment failure risk using Logistic Regression.

This project was completed as part of my learning journey with **LearnDepth Academy LLP**.

The dataset contains operational measurements such as temperature, vibration, pressure, runtime, maintenance gap, and power draw. The target variable indicates whether a failure-risk event is present.

> **Note:** The dataset is synthetic and intended for educational Machine Learning practice.

## Problem Statement

Equipment failure can cause downtime, maintenance costs, and operational disruption.

The objective of this project is to build a binary classification model that predicts equipment failure risk from equipment operating measurements.

## Dataset

**File:** `dataset_06_equipment_failure_warning.csv`

### Input Features

- `temperature`
- `vibration`
- `pressure`
- `runtime_hours`
- `maintenance_gap_days`
- `power_draw`

### Target

- `target = 1` → Failure-risk/event class
- `target = 0` → Non-failure/non-event class

The dataset contains **1,000 rows, 6 input features, and 1 binary target**.

## Machine Learning Workflow

1. Load and inspect the dataset
2. Check missing values and duplicates
3. Check target class balance
4. Perform Exploratory Data Analysis (EDA)
5. Analyze feature relationships
6. Prepare features and target
7. Perform a stratified train-test split
8. Scale numerical features
9. Train Logistic Regression
10. Generate predictions and probabilities
11. Evaluate the model
12. Analyze the confusion matrix and ROC curve
13. Interpret model coefficients
14. Save results and visualizations

## Data Quality

The dataset was checked for:

- Missing values
- Duplicate records
- Target distribution
- Feature ranges
- Data types

No missing values or duplicate records were found, so no rows needed to be removed and no missing-value imputation was required.

Feature scaling was applied because the input variables have different numerical ranges.

## Model

**Algorithm:** Logistic Regression

The model uses a Scikit-learn pipeline containing:

- `StandardScaler`
- `LogisticRegression`

A stratified train-test split was used to maintain the target class distribution.

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 71.50% |
| Precision | 74.16% |
| Recall | 66.00% |
| F1-Score | 69.84% |
| ROC-AUC | 77.88% |

### Confusion Matrix

```text
[[77 23]
 [34 66]]
 