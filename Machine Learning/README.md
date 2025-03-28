# IAA Machine Learning Project

## Machine Learning Coursework – Final Report (Phases 1–3)

This project supports the Commercial Banking Corporation in identifying customers likely to
purchase a variable rate annuity product. It progresses through three modeling phases: initial
modeling (MARS & GAM), tree-based ensemble methods, and model selection with interpretation.

## Objective Summary

### Phase 1: MARS and GAMs (Implemented in R)
- Used the `insurance_t` dataset (8,495 observations).
- Built classification models using:
  - MARS (Multivariate Adaptive Regression Splines) using the `earth` package in R.
  - GAMs (Generalized Additive Models) using `mgcv`, applying splines to continuous predictors in R.
- Compared models using area under the ROC curve (AUC).
- Reported variable importance and visualized model performance.

### Phase 2: Random Forest & XGBoost (Implemented in Python)
- Cleaned and imputed data using median and mode imputation.
- Built and tuned:
  - Random Forest model using `scikit-learn`, with AUC and variable importance analysis.
  - XGBoost model (`objective = "binary:logistic"`) using the `xgboost` library.
- Reported ROC curves and ranked variable importance for both models.

### Phase 3: Model Selection and Interpretation (Implemented in Python)
- Selected final EBM model.
- Reported AUC and ROC on the validation dataset.
- Interpretation included:
  - Global insight using PDP for `ACCTAGE`

## Key Findings
- Tree-based models showed strong predictive power, with EBM performing slightly better in AUC.
- Account age, direct deposit usage, and balance-related features were consistently impactful.
- Interpretation tools revealed both general trends and customer-specific drivers.

## Deliverables
This folder contains:
- Scripts written in R and Python (clearly separated in `code/` folder)
- Plots and visualizations (`visuals/` folder)
- Optional final report PDF (`report/` folder)

## Tools Used
- R packages: `earth`, `mgcv`, `ROCit`, `pdp`, `caret`, `dplyr`, `ggplot2`
- Python packages: `scikit-learn`, `xgboost`, `shap`, `lime`, `matplotlib`, `pandas`, `seaborn`, `interpret`

---

Explore the code and visuals to see how data-driven modeling supports real-world decision-making for customer targeting.
