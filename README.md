# 🚢 Titanic Survival Prediction Using Machine Learning

A complete **Machine Learning classification project** that predicts whether a Titanic passenger survived based on passenger characteristics such as passenger class, gender, age, family information, fare, embarkation port, and cabin availability.

The project includes **Exploratory Data Analysis (EDA), data cleaning, feature engineering, model training, model comparison, evaluation, ROC curve analysis, feature importance, and prediction results**.

---

## 📌 Project Overview

The Titanic dataset is a classic machine learning dataset used to study the factors associated with passenger survival.

This project builds multiple classification models to predict the `Survived` target variable.

The complete pipeline includes:

* Dataset loading
* Exploratory Data Analysis
* Missing-value handling
* Data preprocessing
* Feature engineering
* Train-test splitting
* Machine Learning model training
* Model comparison
* Confusion matrix analysis
* ROC curve comparison
* Feature importance analysis
* Prediction generation
* Classification report generation

---

# 🎯 Objectives

The main objectives of this project are:

1. Analyze the Titanic passenger dataset.
2. Understand passenger survival patterns.
3. Handle missing values.
4. Convert categorical variables into numerical values.
5. Select relevant features for prediction.
6. Train multiple classification algorithms.
7. Compare model performance.
8. Evaluate classification results.
9. Analyze Random Forest feature importance.
10. Generate prediction and evaluation reports.

---

# 🧠 Technologies Used

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| Python              | Programming language      |
| Pandas              | Data manipulation         |
| NumPy               | Numerical operations      |
| Matplotlib          | Data visualization        |
| Seaborn             | Statistical visualization |
| Scikit-learn        | Machine Learning          |
| Logistic Regression | Classification            |
| Decision Tree       | Classification            |
| Random Forest       | Classification            |

---

# 🤖 Machine Learning Models

The project trains and evaluates three classification models.

### 1. Logistic Regression

A linear classification algorithm used as one of the baseline models.

```python
LogisticRegression(max_iter=1000)
```

### 2. Decision Tree

A tree-based classification algorithm that makes predictions through decision rules.

```python
DecisionTreeClassifier(random_state=42)
```

### 3. Random Forest

An ensemble learning algorithm consisting of multiple decision trees.

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

The three models are trained and evaluated using the same train-test split.

---

# 📊 Dataset

The project expects a CSV file named:

```text
Data.csv
```

The dataset should contain Titanic passenger information including the fields required by the model.

Important columns used by the project include:

```text
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
Cabin
Survived
```

Where:

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `Pclass`   | Passenger class                   |
| `Sex`      | Passenger gender                  |
| `Age`      | Passenger age                     |
| `SibSp`    | Number of siblings/spouses aboard |
| `Parch`    | Number of parents/children aboard |
| `Fare`     | Passenger fare                    |
| `Embarked` | Port of embarkation               |
| `Cabin`    | Cabin information                 |
| `Survived` | Target variable                   |

---

# 📂 Project Structure

```text
Titanic-Survival-Prediction/
│
├── Data.csv
│
├── titanic_survival_prediction.py
│
├── requirements.txt
│
├── README.md
│
└── outputs/
    │
    ├── 01_Survival_Distribution.png
    ├── 02_Age_Distribution.png
    ├── 03_Gender_vs_Survival.png
    ├── 04_Pclass_vs_Survival.png
    ├── 05_Correlation_Heatmap.png
    ├── 06_Confusion_Logistic.png
    ├── 07_Confusion_DecisionTree.png
    ├── 08_Confusion_RandomForest.png
    ├── 09_ROC_Curve.png
    ├── 10_Feature_Importance.png
    │
    ├── Processed_Data.csv
    ├── Prediction_Result.csv
    ├── Model_Comparison.csv
    ├── Feature_Importance.csv
    │
    └── Classification_Report.txt
```

---

# 🔄 Project Workflow

```text
                    ┌─────────────────┐
                    │    Data.csv     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Load Dataset   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      EDA        │
                    │ Shape / Info    │
                    │ Missing Values  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Cleaning   │
                    │ Missing Values  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature         │
                    │ Engineering     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Train / Test    │
                    │     80 / 20     │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
          ┌──────────┐ ┌──────────┐ ┌─────────────┐
          │ Logistic │ │ Decision │ │   Random    │
          │Regression│ │  Tree    │ │   Forest    │
          └────┬─────┘ └────┬─────┘ └──────┬──────┘
               │            │              │
               └────────────┼──────────────┘
                            ▼
                   ┌──────────────────┐
                   │ Model Comparison │
                   └────────┬─────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
       ┌────────────┐ ┌───────────┐ ┌──────────────┐
       │ Confusion  │ │ ROC Curve │ │   Feature    │
       │  Matrix    │ │           │ │  Importance  │
       └────────────┘ └───────────┘ └──────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Final Predictions│
                   └──────────────────┘
```

---

# 🔍 Part 1 — Exploratory Data Analysis

The project first loads `Data.csv` and performs basic dataset analysis.

The following information is displayed:

* First five records
* Dataset shape
* Column names
* Dataset information
* Missing values
* Summary statistics

---

# 🧹 Data Cleaning

The project handles missing values in three columns.

### Age

Missing age values are replaced with the median:

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

### Embarked

Missing values are replaced using the most frequent value:

```python
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)
```

### Cabin

Missing cabin values are represented as:

```text
Unknown
```

The cleaned dataset is saved as:

```text
outputs/Processed_Data.csv
```

These cleaning operations are directly implemented in Part 1 of the project.

---

# 📊 Exploratory Data Visualizations

The project generates five EDA visualizations.

### 1. Survival Distribution

```text
01_Survival_Distribution.png
```

Shows the distribution of passengers according to the `Survived` target.

### 2. Age Distribution

```text
02_Age_Distribution.png
```

Shows the distribution of passenger ages.

### 3. Gender vs Survival

```text
03_Gender_vs_Survival.png
```

Compares survival outcomes across passenger genders.

### 4. Passenger Class vs Survival

```text
04_Pclass_vs_Survival.png
```

Shows survival distribution across passenger classes.

### 5. Correlation Heatmap

```text
05_Correlation_Heatmap.png
```

Displays correlations between numerical variables.

The five visualizations are generated during Part 1.

---

# ⚙️ Feature Engineering

Categorical variables are converted into numerical representations.

### Sex

The `Sex` column is encoded using `LabelEncoder`.

### Embarked

The `Embarked` column is also encoded using `LabelEncoder`.

### Cabin

Instead of using the actual cabin value, the project converts it into a binary feature:

```text
0 → Cabin Unknown
1 → Cabin Known
```

The model uses these eight features:

```text
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
Cabin
```

The target variable is:

```text
Survived
```

This feature-selection and encoding process is implemented in Part 2.

---

# ✂️ Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The project uses:

```python
random_state=42
```

and stratified sampling to maintain the target-class distribution between the training and testing sets.

---

# 📈 Model Evaluation Metrics

The following metrics are calculated for each model:

### Accuracy

Measures the proportion of correct predictions.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

Combines precision and recall into a single metric.

### ROC AUC

Measures classification performance based on predicted probabilities.

The project stores these metrics in:

```text
outputs/Model_Comparison.csv
```

The code calculates all five metrics for Logistic Regression, Decision Tree, and Random Forest.

---

# 📋 Model Comparison

The generated comparison file has the following structure:

```text
Model
Accuracy
Precision
Recall
F1 Score
ROC AUC
```

Example:

```text
Model                  Accuracy    Precision    Recall    F1 Score    ROC AUC
Logistic Regression       ...
Decision Tree             ...
Random Forest             ...
```

The actual values are generated when the project is executed.

---

# 📉 Confusion Matrix Analysis

Three confusion matrices are generated.

### Logistic Regression

```text
06_Confusion_Logistic.png
```

### Decision Tree

```text
07_Confusion_DecisionTree.png
```

### Random Forest

```text
08_Confusion_RandomForest.png
```

Each confusion matrix shows:

```text
Not Survived
Survived
```

as the two classification categories.

---

# 📈 ROC Curve Comparison

The project generates:

```text
09_ROC_Curve.png
```

The ROC curves for all three models are plotted together:

```text
Logistic Regression
Decision Tree
Random Forest
```

This allows their classification behavior to be visually compared.

---

# 🌲 Random Forest Feature Importance

The Random Forest model provides feature-importance values for the selected features.

The importance values are stored in:

```text
outputs/Feature_Importance.csv
```

A visualization is generated as:

```text
10_Feature_Importance.png
```

The features are sorted according to their Random Forest importance values.

---

# 🔮 Prediction Results

The Random Forest model is used to generate predictions for the test dataset.

The results are stored in:

```text
outputs/Prediction_Result.csv
```

The output contains the test features along with:

```text
Actual
Predicted
```

This provides a direct comparison between actual and predicted survival outcomes.

---

# 📄 Classification Report

A classification report is generated using the Random Forest predictions.

The report is saved as:

```text
outputs/Classification_Report.txt
```

It contains the classification performance information for the Random Forest model.

---

# 📁 Generated Outputs

After running the project, the following files are generated:

| File                            | Description                          |
| ------------------------------- | ------------------------------------ |
| `01_Survival_Distribution.png`  | Survival distribution                |
| `02_Age_Distribution.png`       | Passenger age distribution           |
| `03_Gender_vs_Survival.png`     | Gender vs survival                   |
| `04_Pclass_vs_Survival.png`     | Passenger class vs survival          |
| `05_Correlation_Heatmap.png`    | Feature correlation heatmap          |
| `06_Confusion_Logistic.png`     | Logistic Regression confusion matrix |
| `07_Confusion_DecisionTree.png` | Decision Tree confusion matrix       |
| `08_Confusion_RandomForest.png` | Random Forest confusion matrix       |
| `09_ROC_Curve.png`              | ROC curve comparison                 |
| `10_Feature_Importance.png`     | Random Forest feature importance     |
| `Processed_Data.csv`            | Cleaned dataset                      |
| `Prediction_Result.csv`         | Model predictions                    |
| `Model_Comparison.csv`          | Model evaluation comparison          |
| `Feature_Importance.csv`        | Feature importance values            |
| `Classification_Report.txt`     | Random Forest classification report  |

The code explicitly generates these files at the end of execution.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/syedshaju53/Titanic-Survival-Prediction.git
```

Move into the project directory:

```bash
cd Titanic-Survival-Prediction
```

---

# 🐍 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

# 📦 3. Install Dependencies

Create a file named:

```text
requirements.txt
```

Add:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Then install:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Make sure the project contains:

```text
Data.csv
titanic_survival_prediction.py
```

Then run:

### macOS / Linux

```bash
python3 titanic_survival_prediction.py
```

### Windows

```bash
python titanic_survival_prediction.py
```

The program automatically creates the:

```text
outputs/
```

directory if it does not already exist.

---

# 🧪 Expected Execution Flow

```text
Load Dataset
     ↓
EDA
     ↓
Handle Missing Values
     ↓
Save Processed Dataset
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Train 3 Models
     ↓
Model Comparison
     ↓
Confusion Matrices
     ↓
ROC Curve
     ↓
Feature Importance
     ↓
Generate Predictions
     ↓
Classification Report
     ↓
Project Completed
```

At the end, the terminal displays:

```text
============================================================
PROJECT COMPLETED SUCCESSFULLY
============================================================
```

---

# 🚀 Future Improvements

Possible extensions for this project include:

* Hyperparameter tuning
* Cross-validation
* GridSearchCV
* RandomizedSearchCV
* XGBoost
* LightGBM
* Gradient Boosting
* Feature engineering such as family size
* Age groups
* Fare groups
* Passenger title extraction
* Streamlit prediction dashboard
* Model deployment using FastAPI
* Model serialization using Joblib
* Interactive passenger survival prediction

---

# ⚠️ Note

This project is intended for **educational and machine-learning demonstration purposes**.

The model learns patterns from the provided dataset and its predictions depend on the quality and characteristics of that dataset.

---

# 🎓 Project Information

**Project:** Titanic Survival Prediction Using Machine Learning

**Domain:** Data Science / Machine Learning

**Problem Type:** Binary Classification

**Target Variable:**

```text
Survived
```

**Models Used:**

```text
Logistic Regression
Decision Tree
Random Forest
```

**Evaluation Metrics:**

```text
Accuracy
Precision
Recall
F1 Score
ROC AUC
```

---

# 👨‍💻 Author

**Syed Shajahan**

BTech — Data Science

---

## ⭐ GitHub

If you find this project useful, consider giving the repository a ⭐.
