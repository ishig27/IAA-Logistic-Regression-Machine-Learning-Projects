# ---------------------------
# Phase 2 - Random Forest Modeling & XGBoost Modeling
# ---------------------------

# Importing necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance
import shap
import warnings

# Load datasets
train = pd.read_csv("~/GitHub/IAA-Data-Science-projects/Machine Learning/data/insurance_t.csv")
valid = pd.read_csv("~/GitHub/IAA-Data-Science-projects/Machine Learning/data/insurance_v.csv")

# Prepare and clean the data
# Identify categorical variables (<=10 unique values)
factor_cols = ["DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "INV",
               "MM", "MMCRED", "CC", "CCPURC", "SDB", "INAREA"]

train[factor_cols] = train[factor_cols].astype("category")
valid[factor_cols] = valid[factor_cols].astype("category")

# Drop 'INS' column for imputation
train_imp = train.drop(columns=["INS"])
valid_imp = valid.drop(columns=["INS"])

# Define imputers
numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

# Separate columns by type
numeric_cols = train_imp.select_dtypes(include=np.number).columns
categorical_cols = train_imp.select_dtypes(include="category").columns

# Impute missing values in training data
train_imp[numeric_cols] = numeric_imputer.fit_transform(train_imp[numeric_cols])
train_imp[categorical_cols] = categorical_imputer.fit_transform(train_imp[categorical_cols])

# Impute missing values in validation data
valid_imp[numeric_cols] = numeric_imputer.transform(valid_imp[numeric_cols])
valid_imp[categorical_cols] = categorical_imputer.transform(valid_imp[categorical_cols])

# Convert imputed categorical columns back to category type
train_imp[categorical_cols] = train_imp[categorical_cols].apply(lambda x: x.astype("category"))
valid_imp[categorical_cols] = valid_imp[categorical_cols].apply(lambda x: x.astype("category"))

# Add back 'INS' target variable
train_imp["INS"] = train["INS"]
valid_imp["INS"] = valid["INS"]

# One-hot encode categorical variables
X_train = pd.get_dummies(train_imp.drop(columns="INS"), drop_first=True)
X_valid = pd.get_dummies(valid_imp.drop(columns="INS"), drop_first=True)

# Align columns to ensure consistency
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
y_train = train_imp["INS"]

# ---------------------------
# Random Forest Modeling
# ---------------------------

# Encode categorical features for Random Forest
train_encoded = pd.get_dummies(train_imp.drop(columns="INS"), drop_first=True)
train_encoded["INS"] = train_imp["INS"]

RANDOM_STATE = 1234

# Initial Random Forest
rf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, oob_score=True)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.nlargest(10).sort_values().plot(kind='barh', title="Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# OOB Error Plot
errors = []
for i in range(10, 260, 10):
    rf_temp = RandomForestClassifier(n_estimators=i, random_state=RANDOM_STATE, oob_score=True)
    rf_temp.fit(X_train, y_train)
    errors.append(1 - rf_temp.oob_score_)

plt.plot(range(10, 260, 10), errors, marker='o')
plt.title("OOB Error vs. Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("OOB Error")
plt.grid(True)
plt.show()

# Tuned Random Forest with max_features
rf_tuned = RandomForestClassifier(n_estimators=250, max_features=5, random_state=RANDOM_STATE)
rf_tuned.fit(X_train, y_train)

importances_tuned = pd.Series(rf_tuned.feature_importances_, index=X_train.columns)
importances_tuned.nlargest(10).sort_values().plot(kind='barh', title="Top 10 Feature Importances (mtry=5)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Add random feature
np.random.seed(RANDOM_STATE)
X_train["random"] = np.random.normal(size=len(X_train))

rf_with_random = RandomForestClassifier(n_estimators=250, max_features=5, random_state=RANDOM_STATE)
rf_with_random.fit(X_train, y_train)

importances_random = pd.Series(rf_with_random.feature_importances_, index=X_train.columns)
importances_random.nlargest(10).sort_values().plot(kind='barh', title="Feature Importance Including Random Feature")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Reseed and rerun
np.random.seed(789)
X_train["random"] = np.random.normal(size=len(X_train))

rf_consistency = RandomForestClassifier(n_estimators=250, max_features=6, random_state=789)
rf_consistency.fit(X_train, y_train)

importances_consistency = pd.Series(rf_consistency.feature_importances_, index=X_train.columns)
importances_consistency.nlargest(10).sort_values().plot(kind='barh', title="Feature Importance (Different Seed)")
plt.tight_layout()
plt.show()


# --- Final RF on selected features only ---
# Train a final Random Forest model using top-selected features
# SAVBAL: Savings Balance
# BRANCH: Branch Identifier (categorical)
# DDABAL: DDA Account Balance

selected_features = ["SAVBAL", "BRANCH", "DDABAL"]
new_data = pd.concat([train["INS"], pd.get_dummies(train[selected_features], drop_first=True)], axis=1)

# Add slight regularization to prevent overfitting
rf_final = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=789)
rf_final.fit(new_data.drop(columns="INS"), new_data["INS"])

# Plot final model importance
final_importances = pd.Series(rf_final.feature_importances_, index=new_data.drop(columns="INS").columns)
final_importances.sort_values().plot(kind='barh', title="Final Model Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# --- ROC Curve and Discrimination Analysis ---
# Evaluate model performance using AUC and ROC curve
# Calculate optimal classification threshold using Youden's J statistic

# Predictions
new_data["predict_INS_rf"] = rf_final.predict_proba(new_data.drop(columns="INS"))[:, 1]

# AUC
auc = roc_auc_score(new_data["INS"], new_data["predict_INS_rf"])
print("Area Under the ROC Curve (AUC):", round(auc, 4))

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(new_data["INS"], new_data["predict_INS_rf"])
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

# Optimal threshold (Youden's J statistic)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold:", round(optimal_threshold, 4))

# Coefficient of Discrimination
rf_p1 = new_data[new_data["INS"] == 1]["predict_INS_rf"]
rf_p0 = new_data[new_data["INS"] == 0]["predict_INS_rf"]
coef_discrim = rf_p1.mean() - rf_p0.mean()
print("Coefficient of Discrimination:", round(coef_discrim, 4))

# Discrimination Plot
plt.figure()
for label in [0, 1]:
    subset = new_data[new_data["INS"] == label]
    plt.hist(subset["predict_INS_rf"], bins=50, alpha=0.5, label=f"INS = {label}", density=True)
plt.axvline(optimal_threshold, color='red', linestyle='dashed', label="Optimal Cutoff")
plt.title("Random Forest Discrimination Plot")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# XGBoost Modeling
# ---------------------------

# --- XGBoost Hyperparameter Tuning ---

# Encode categorical variables
X_train_full = pd.get_dummies(train_imp.drop(columns=["INS"]), drop_first=True)
y_train_full = train_imp["INS"]

# Parameter grid for tuning
param_grid = {
    'n_estimators': [100],
    'eta': [0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': list(range(1, 11)),
    'gamma': [0],
    'colsample_bytree': [1],
    'min_child_weight': [1],
    'subsample': [0.25, 0.5, 0.75, 1]
}

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False, random_state=123)

# Perform cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
grid_search = GridSearchCV(xgb_model, param_grid, scoring="roc_auc", cv=cv, verbose=1)
grid_search.fit(X_train_full, y_train_full)

# Best hyperparameters
print("Best parameters:", grid_search.best_params_)

# Variable Importance Test with Random Variable 
X_train_full["random"] = np.random.normal(size=len(X_train_full))
xgb_model_random = xgb.XGBClassifier(**grid_search.best_params_, objective='binary:logistic', eval_metric='auc', random_state=123)
xgb_model_random.fit(X_train_full, y_train_full)

# Plot importance including random
plt.figure()
plot_importance(xgb_model_random, max_num_features=10)
plt.title("XGBoost Feature Importance (Random Variable Included)")
plt.tight_layout()
plt.show()

# --- Final XGBoost Model with Selected Features ---

# Select features based on prior analysis
xgb_features = ["SAVBAL", "DDABAL", "DDA", "CDBAL", "MMBAL"]

# Encode categorical variables using one-hot encoding
X_train = pd.get_dummies(train_imp[xgb_features], drop_first=True)
y_train = train_imp["INS"]

# Define XGBoost classifier with tuned hyperparameters
xgb_final = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=3,                # Limits tree depth to prevent overfitting
    eta=0.1,                    # Learning rate
    gamma=0,                    # Minimum loss reduction for further partitioning
    subsample=0.75,             # Row subsampling ratio
    colsample_bytree=1,         # Column subsampling per tree
    min_child_weight=1,         # Minimum sum of instance weight in a child
    n_estimators=100,           # Number of boosting rounds
    use_label_encoder=False,
    eval_metric='auc',
    random_state=123
)

# Train the model
xgb_final.fit(X_train, y_train)

# Plot top 10 most important features (gain-based)
plt.figure()
plot_importance(xgb_final, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Predict probabilities on training set
train_imp["xgb_pred"] = xgb_final.predict_proba(X_train)[:, 1]



# --- ROC Curve, AUC and Discrimination Analysis ---

# ROC AUC Score
auc = roc_auc_score(y_train, train_imp["xgb_pred"])
print("Area Under the ROC Curve (AUC):", round(auc, 4))

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_train, train_imp["xgb_pred"])
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.tight_layout()
plt.show()

# Identify optimal cutoff using Youden's J statistic
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold:", round(optimal_threshold, 4))

# Coefficient of Discrimination
xgb_p1 = train_imp[train_imp["INS"] == 1]["xgb_pred"]
xgb_p0 = train_imp[train_imp["INS"] == 0]["xgb_pred"]
coef_discrim = xgb_p1.mean() - xgb_p0.mean()
print("Coefficient of Discrimination:", round(coef_discrim, 4))

# Discrimination Plot
plt.figure()
for label in [0, 1]:
    subset = train_imp[train_imp["INS"] == label]
    plt.hist(subset["xgb_pred"], bins=50, alpha=0.5, label=f"INS = {label}", density=True)
plt.axvline(optimal_threshold, color='red', linestyle='dashed', label="Optimal Cutoff")
plt.title("XGBoost Discrimination Plot")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

