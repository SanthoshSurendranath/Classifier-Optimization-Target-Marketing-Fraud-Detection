# Classifier Optimization — Insurance Fraud Detection & Bank Target Marketing

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?logo=scikit-learn) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project implements and optimizes classification models across two distinct business domains: **insurance fraud detection** and **bank term deposit target marketing**. Both use cases follow an identical machine learning pipeline — baseline classification, hyperparameter tuning via Randomized and Grid Search CV, and performance evaluation — enabling a direct comparison of classifier behaviour across different data characteristics and class imbalance levels.

The project is structured in two independent parts:

- **Part A** — Insurance Fraud Detection (binary classification on imbalanced claim data)
- **Part B** — Bank Term Deposit Subscription Prediction (binary classification for marketing campaign optimization)

---

## Business Objectives

**Part A — Insurance Fraud Detection**
Identify fraudulent auto insurance claims from policyholder and accident attributes, enabling insurers to flag high-risk claims for investigation and reduce financial losses from fraud.

**Part B — Bank Target Marketing**
Predict which bank customers are likely to subscribe to a term deposit product based on demographic and campaign interaction data, allowing marketing teams to prioritize outreach and improve conversion rates.

---

## Datasets

### Part A — Insurance Fraud

| Dataset | Records | Features | Target | Source |
|---|---|---|---|---|
| `Insurance Fraud - TRAIN-3000.csv` | 2,999 | 32 | `FRAUDFOUND` (Yes/No) | Google Drive |
| `Insurance Fraud - TEST-12900.csv` | 12,918 | 32 | `FRAUDFOUND` (Yes/No) | Google Drive |

**Key Features:** Month, Day of Week, Vehicle Make, Accident Area, Policy Type, Vehicle Category, Vehicle Price, Driver Rating, Agent Type, Base Policy, and 20+ additional claim and policyholder attributes.

**Class Imbalance Note:** Fraud cases represent a minority class (~3.9% of test records), making precision-recall tradeoff critical.

---

### Part B — Bank Target Marketing (Portuguese Bank)

| Dataset | Records | Features | Target | Source |
|---|---|---|---|---|
| `Portugese Bank Data - TRAIN.csv` | 4,521 | 17 | `y` (yes/no) | Google Drive |
| `Portugese Bank Data - TEST.csv` | 45,211 | 17 | `y` (yes/no) | Google Drive |

**Key Features:** Age, Job, Marital Status, Education, Balance, Housing Loan, Contact Type, Month, Campaign Duration, Previous Outcome, and related banking attributes.

**Class Imbalance Note:** Term deposit subscriptions represent ~11.7% of the test set, requiring careful metric selection beyond raw accuracy.

---

## Project Structure

```
ClassifierOptimization/
│
├── Classifier-Optimization-Target-Marketing-Fraud-Detection Part - A.ipynb   # Insurance Fraud Pipeline
├── Classifier-Optimization-Target-Marketing-Fraud-Detection Part - B.ipynb   # Bank Marketing Pipeline
└── README.md
```

---

## Methodology

Both notebooks follow the same structured pipeline:

### 1. Data Loading & Exploration
- Dataset shapes, column types, and null value inspection via `.info()`
- Feature and target separation

### 2. Feature Engineering
- One-Hot Encoding applied to all categorical features using `sklearn.preprocessing.OneHotEncoder`
- Encoder fit on training data only; test data transformed using the fitted encoder to prevent data leakage
- Post-encoding feature space: **139 columns** (Part A) | **51 columns** (Part B)

### 3. Baseline Modelling
- Decision Tree Classifier (default parameters)
- Random Forest Classifier (default parameters)
- Metrics reported: Accuracy, Confusion Matrix, Classification Report (Precision, Recall, F1), Feature Importances

### 4. Hyperparameter Tuning
Both RandomizedSearchCV and GridSearchCV were applied across multiple parameter grids:

**Decision Tree Parameters Tuned:**
- `criterion`: gini, entropy
- `max_depth`: varied ranges (5–30)
- `min_samples_leaf`: varied ranges (10–700)

**Random Forest Parameters Tuned:**
- `n_estimators`: 20, 30, 40
- `max_depth`: 1–9
- `max_features`: 10, 20, 30
- `min_samples_leaf`: 10–100

### 5. Model Evaluation
- Cross-validation with `balanced_accuracy` scoring to account for class imbalance
- Final tuned models compared on held-out test sets

---

## Results Summary

### Part A — Insurance Fraud Detection

| Model | Configuration | Test Accuracy | Notes |
|---|---|---|---|
| Decision Tree | Default | 88.3% | Overfits training (100% train acc); fraud recall 90% |
| Random Forest | Default | **96.6%** | Best baseline; fraud precision 54%, recall 84% |
| Decision Tree | RandomizedSearchCV | 88.3% | Reduced overfitting; fraud precision drops |
| Decision Tree | GridSearchCV (Metric 2) | **89.7%** | Best tuned DT; entropy, depth=7, leaf=30 |
| Random Forest | RandomizedSearchCV | 92.9% | n_estimators=40, depth=5, features=30 |
| Random Forest | GridSearchCV | 92.7% | n_estimators=30, depth=9, features=30 |

> **Key Insight:** Default Random Forest achieved the highest test accuracy but at the cost of low fraud precision (54%). Tuned models trade some accuracy for better generalisation, as confirmed by 5-fold balanced cross-validation scores.

---

### Part B — Bank Term Deposit Prediction

| Model | Configuration | Test Accuracy | Notes |
|---|---|---|---|
| Decision Tree | Default | 88.1% | Overfits training (100% train acc); subscription recall 52% |
| Random Forest | Default | 91.1% | Subscription precision 75%, recall 36% |
| Decision Tree | RandomizedSearchCV | 89.7% | depth=10, leaf=25, entropy |
| Decision Tree | GridSearchCV (Metric 2) | 89.6% | depth=5, leaf=30, entropy |
| Random Forest | RandomizedSearchCV | 90.0% | n_estimators=30, depth=5, features=30 |
| Random Forest | GridSearchCV | **90.1%** | Best tuned RF; n_estimators=20, depth=9, features=30 |

> **Key Insight:** Tuned Random Forest achieved the best balance between accuracy and generalisation. Top predictive features include call duration, previous campaign outcome, and account balance. Balanced accuracy cross-validation (~64%) confirms the challenge posed by class imbalance.

---

## Technologies & Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Classification models, encoding, tuning, evaluation |
| `Google Colab` | Cloud-based notebook execution environment |
| `Google Drive` | Dataset storage and I/O |

---

## Setup & Usage

### Prerequisites

```bash
pip install scikit-learn pandas numpy
```

### Running the Notebooks

1. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/gdrive')
   ```

2. Place the following files in your Google Drive root:
   - `Insurance Fraud - TRAIN-3000.csv`
   - `Insurance Fraud -TEST-12900.csv`
   - `Portugese Bank Data - TRAIN.csv`
   - `Portugese Bank Data - TEST.csv`

3. Run **Part A** for fraud detection modelling.
4. Run **Part B** for bank marketing modelling.

---

## Key Findings

- **Random Forest consistently outperforms Decision Tree** across both datasets in baseline configuration, but at the cost of interpretability.
- **Hyperparameter tuning reduces overfitting** significantly — default Decision Trees achieved 100% training accuracy on both datasets, a clear sign of overfitting.
- **Class imbalance is a shared challenge** across both use cases. Raw accuracy is a misleading metric; precision-recall and balanced accuracy provide a more reliable picture of model quality.
- **GridSearchCV is more exhaustive but computationally expensive** — Part A Grid Search took 211 seconds vs. 21 seconds for RandomizedSearchCV, with only marginal accuracy gains.
- The **entropy criterion** with shallow-to-moderate tree depths consistently produced better-generalising Decision Trees than gini in these datasets.

---

## Author

Santhosh Surendranath
Data Scientist

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

