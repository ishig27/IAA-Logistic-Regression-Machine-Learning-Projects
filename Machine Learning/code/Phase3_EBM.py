# -----------------------------
# Phase 3 - Explainable Boosting Machine (EBM)
# -----------------------------

# Libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show, set_visualize_provider
from interpret.provider import InlineProvider
import matplotlib.pyplot as plt
import skexplain
from sklearn.ensemble import RandomForestClassifier

# --- Data Loading and Preparation ---

train = pd.read_csv("~/GitHub/IAA-Data-Science-projects/Machine Learning/data/insurance_t.csv")
valid = pd.read_csv("~/GitHub/IAA-Data-Science-projects/Machine Learning/data/insurance_v.csv")

factor_cols = ["DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "INV",
               "MM", "MMCRED", "CC", "CCPURC", "SDB", "INAREA"]

# Set categorical columns
train[factor_cols] = train[factor_cols].astype("category")
valid[factor_cols] = valid[factor_cols].astype("category")

# Drop 'INS' temporarily for imputation
train_imp, valid_imp = train.drop(columns=["INS"]), valid.drop(columns=["INS"])

# Imputation
numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

num_cols = train_imp.select_dtypes(include=np.number).columns
cat_cols = train_imp.select_dtypes(include="category").columns

train_imp[num_cols] = numeric_imputer.fit_transform(train_imp[num_cols])
train_imp[cat_cols] = categorical_imputer.fit_transform(train_imp[cat_cols])
valid_imp[num_cols] = numeric_imputer.transform(valid_imp[num_cols])
valid_imp[cat_cols] = categorical_imputer.transform(valid_imp[cat_cols])

train_imp[cat_cols] = train_imp[cat_cols].astype("category")
valid_imp[cat_cols] = valid_imp[cat_cols].astype("category")

# Add 'INS' back
train_imp["INS"], valid_imp["INS"] = train["INS"], valid["INS"]

# --- EBM Model Training & Evaluation ---
X_train = pd.get_dummies(train_imp.drop(columns="INS"), drop_first=True)
y_train = train_imp["INS"].astype(int)

X_valid = pd.get_dummies(valid_imp.drop(columns="INS"), drop_first=True)
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
y_valid = valid_imp["INS"].astype(int)

np.random.seed(1234)
eb_model = ExplainableBoostingClassifier(interactions=2)
eb_model.fit(X_train, y_train)

# ROC Curve pre variable selection
valid_preds = eb_model.predict_proba(X_valid)[:, 1]
roc_auc = roc_auc_score(y_valid, valid_preds)
print(f"ROC-AUC: {roc_auc:.5f}")

fpr, tpr, thresholds = roc_curve(y_valid, valid_preds)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
accuracy = accuracy_score(y_valid, valid_preds >= optimal_threshold)
print(f"Accuracy: {accuracy:.5f}")

plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - EBM')
plt.legend()
plt.grid()
plt.show()

# --- Global Interpretability ---
set_visualize_provider(InlineProvider())
eb_global = eb_model.explain_global()
show(eb_global)
# Interpretation:
# SAVBAL significantly influences predictions, with about 40% impact on the model predictions.

# --- Variable Importance Check ---
X_train_random = X_train.copy()
X_train_random['random'] = np.random.normal(size=X_train.shape[0])

eb_random = ExplainableBoostingClassifier(interactions=2)
eb_random.fit(X_train_random, y_train)

importances_df = pd.DataFrame({
    "Variable": eb_random.term_names_,
    "Importance": eb_random.term_importances()
}).sort_values(by="Importance", ascending=False)

random_importance = importances_df[importances_df.Variable == "random"].Importance.iloc[0]
selected_vars = importances_df[importances_df.Importance > random_importance].Variable.tolist()

selected_vars = [var for var in selected_vars if var != "random"] + ['INS']

# Extract only original variable names (no interactions or encoded categories)
selected_vars = importances_df[
    (importances_df.Importance > random_importance) &
    (~importances_df.Variable.str.contains('&')) &
    (~importances_df.Variable.str.contains('_'))
].Variable.tolist()

selected_vars = [var for var in selected_vars if var != "random"] + ['INS']

train_subset, valid_subset = train_imp[selected_vars], valid_imp[selected_vars]

train_subset, valid_subset = train_imp[selected_vars], valid_imp[selected_vars]

# --- Retrain EBM with Selected Variables --- 
X_train_sub = pd.get_dummies(train_subset.drop(columns="INS"), drop_first=True)
y_train_sub = train_subset["INS"].astype(int)

X_valid_sub = pd.get_dummies(valid_subset.drop(columns="INS"), drop_first=True)
X_valid_sub = X_valid_sub.reindex(columns=X_train_sub.columns, fill_value=0)
y_valid_sub = valid_subset["INS"].astype(int)

np.random.seed(1234)
eb_final = ExplainableBoostingClassifier(interactions=2)
eb_final.fit(X_train_sub, y_train_sub)

# Re-evaluate on subset
final_preds = eb_final.predict_proba(X_valid_sub)[:, 1]
final_roc_auc = roc_auc_score(y_valid_sub, final_preds)
print(f"Final ROC-AUC: {final_roc_auc:.5f}")


#  --- ALE Interpretation (Account Age) --- 
rf = RandomForestClassifier(n_estimators=100, random_state=12345, max_features=7)
rf.fit(X_train_sub, y_train_sub)

explainer = skexplain.ExplainToolkit(('Random Forest', rf), X_train_sub, y_train_sub)
ale_result = explainer.ale(features=['ACCTAGE'])

explainer.plot_ale(ale=ale_result, figsize=(8, 4))

# ALE Interpretation for Account Age:
# - Younger accounts are less likely to purchase insurance; marketing efforts should focus here.
# - Older accounts exhibit higher likelihood of insurance purchase; targeted campaigns or products are recommended.
