


# ==========================================================
# TITANIC SURVIVAL PREDICTION USING MACHINE LEARNING
# PART 1
# Load Dataset
# EDA
# Data Cleaning
# Save Cleaned Dataset
# ==========================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

csv_file = BASE_DIR / "Data.csv"

print("=" * 60)
print("Loading Titanic Dataset")
print("=" * 60)

df = pd.read_csv(csv_file)

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# BASIC EDA
# ==========================================================

print("\nSummary Statistics")
print(df.describe(include="all"))

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

# Age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin
df["Cabin"] = df["Cabin"].fillna("Unknown")

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==========================================================
# SAVE CLEANED DATASET
# ==========================================================

cleaned_file = OUTPUT_DIR / "Processed_Data.csv"

df.to_csv(cleaned_file, index=False)

print("\nProcessed Dataset Saved")

# ==========================================================
# GRAPH 1
# Survival Distribution
# ==========================================================

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"01_Survival_Distribution.png")
plt.close()

# ==========================================================
# GRAPH 2
# Age Distribution
# ==========================================================

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"02_Age_Distribution.png")
plt.close()

# ==========================================================
# GRAPH 3
# Gender vs Survival
# ==========================================================

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Gender vs Survival")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"03_Gender_vs_Survival.png")
plt.close()

# ==========================================================
# GRAPH 4
# Passenger Class vs Survival
# ==========================================================

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Passenger Class vs Survival")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"04_Pclass_vs_Survival.png")
plt.close()

# ==========================================================
# GRAPH 5
# Correlation Heatmap
# ==========================================================

numeric = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"05_Correlation_Heatmap.png")
plt.close()

# ==========================================================
# DISPLAY CLEANED DATA
# ==========================================================

print("\nFirst Five Rows After Cleaning")
print(df.head())

print("\nProcessed Dataset Location")
print(cleaned_file)

print("\nGenerated Files")

print("""
01_Survival_Distribution.png
02_Age_Distribution.png
03_Gender_vs_Survival.png
04_Pclass_vs_Survival.png
05_Correlation_Heatmap.png
Processed_Data.csv
""")

print("="*60)
print("PART 1 COMPLETED SUCCESSFULLY")
print("="*60)

# ==========================================================
# PART 2
# FEATURE ENGINEERING & MODEL TRAINING
# ==========================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*60)
print("PART 2 : MODEL TRAINING")
print("="*60)

# ==========================================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================================

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])

df["Embarked"] = le.fit_transform(df["Embarked"])

# Cabin only indicates known/unknown
df["Cabin"] = df["Cabin"].apply(lambda x: 0 if x == "Unknown" else 1)

# ==========================================================
# SELECT FEATURES
# ==========================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Cabin"
]

X = df[features]
y = df["Survived"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# CREATE MODELS
# ==========================================================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

results = []

trained_models = {}

# ==========================================================
# TRAIN & EVALUATE
# ==========================================================

for name, model in models.items():

    model.fit(X_train, y_train)

    trained_models[name] = model

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    roc = roc_auc_score(y_test, probability)

    results.append({
        "Model":name,
        "Accuracy":round(accuracy,4),
        "Precision":round(precision,4),
        "Recall":round(recall,4),
        "F1 Score":round(f1,4),
        "ROC AUC":round(roc,4)
    })

# ==========================================================
# SAVE MODEL COMPARISON
# ==========================================================

comparison = pd.DataFrame(results)

comparison_file = OUTPUT_DIR / "Model_Comparison.csv"

comparison.to_csv(
    comparison_file,
    index=False
)

print("\nModel Comparison")
print(comparison)

print("\nSaved Successfully")
print(comparison_file)

print("\nPART 2 COMPLETED SUCCESSFULLY")
# ==========================================================
# PART 3
# MODEL EVALUATION & FINAL DELIVERABLES
# ==========================================================

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay
)

print("\n" + "="*60)
print("PART 3 : MODEL EVALUATION")
print("="*60)

# ----------------------------------------------------------
# Confusion Matrix - Logistic Regression
# ----------------------------------------------------------

log_model = trained_models["Logistic Regression"]
log_pred = log_model.predict(X_test)

cm = confusion_matrix(y_test, log_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived","Survived"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression")
plt.savefig(OUTPUT_DIR/"06_Confusion_Logistic.png")
plt.close()

# ----------------------------------------------------------
# Confusion Matrix - Decision Tree
# ----------------------------------------------------------

tree_model = trained_models["Decision Tree"]
tree_pred = tree_model.predict(X_test)

cm = confusion_matrix(y_test, tree_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived","Survived"]
)

disp.plot(cmap="Greens")
plt.title("Decision Tree")
plt.savefig(OUTPUT_DIR/"07_Confusion_DecisionTree.png")
plt.close()

# ----------------------------------------------------------
# Confusion Matrix - Random Forest
# ----------------------------------------------------------

rf_model = trained_models["Random Forest"]
rf_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived","Survived"]
)

disp.plot(cmap="Oranges")
plt.title("Random Forest")
plt.savefig(OUTPUT_DIR/"08_Confusion_RandomForest.png")
plt.close()

# ----------------------------------------------------------
# ROC CURVE
# ----------------------------------------------------------

plt.figure(figsize=(7,6))

RocCurveDisplay.from_estimator(
    log_model,
    X_test,
    y_test,
    name="Logistic Regression"
)

RocCurveDisplay.from_estimator(
    tree_model,
    X_test,
    y_test,
    name="Decision Tree"
)

RocCurveDisplay.from_estimator(
    rf_model,
    X_test,
    y_test,
    name="Random Forest"
)

plt.title("ROC Curve Comparison")
plt.savefig(OUTPUT_DIR/"09_ROC_Curve.png")
plt.close()

# ----------------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------------

importance = pd.DataFrame({
    "Feature":features,
    "Importance":rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR/"Feature_Importance.csv",
    index=False
)

plt.figure(figsize=(8,5))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"10_Feature_Importance.png")
plt.close()

# ----------------------------------------------------------
# SAVE PREDICTIONS
# ----------------------------------------------------------

prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values
prediction_df["Predicted"] = rf_pred

prediction_df.to_csv(
    OUTPUT_DIR/"Prediction_Result.csv",
    index=False
)

# ----------------------------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------------------------

report = classification_report(
    y_test,
    rf_pred
)

with open(
    OUTPUT_DIR/"Classification_Report.txt",
    "w"
) as f:

    f.write("TITANIC SURVIVAL PREDICTION\n")
    f.write("="*50)
    f.write("\n\n")

    f.write("Random Forest Classification Report\n\n")
    f.write(report)

print("\nClassification Report")
print(report)

# ----------------------------------------------------------
# FINAL OUTPUT
# ----------------------------------------------------------

print("\n")
print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("\nFiles Generated:")

print("""
01_Survival_Distribution.png
02_Age_Distribution.png
03_Gender_vs_Survival.png
04_Pclass_vs_Survival.png
05_Correlation_Heatmap.png
06_Confusion_Logistic.png
07_Confusion_DecisionTree.png
08_Confusion_RandomForest.png
09_ROC_Curve.png
10_Feature_Importance.png

Processed_Data.csv
Prediction_Result.csv
Model_Comparison.csv
Feature_Importance.csv

Classification_Report.txt
""")

print("Titanic Survival Prediction Completed Successfully!")