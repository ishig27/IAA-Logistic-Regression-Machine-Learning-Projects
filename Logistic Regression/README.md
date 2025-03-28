# IAA Logisitc Regression Project

## Logistic Regression Coursework – Final Report (Phases 1–3)

### Project Overview
This project supports the Commercial Banking Corporation in identifying customers likely to purchase a variable rate annuity product. It progresses through three phases: understanding variables, building a model, and assessing predictive performance.

### Objective Summary

#### Phase 1: Variable Understanding and Assumptions
- Used the training dataset (`insurance_t`, 8,495 observations).
- Explored individual predictors with the target variable `INS` (1 = bought insurance).
- Identified significant predictors across four types: binary, ordinal, nominal, and continuous.
- Ranked variables by significance and summarized key insights.
- Calculated and interpreted odds ratios for binary variables.
- Evaluated linearity assumptions for continuous predictors.
- Highlighted variables with high missing values (with visualization) and noted redundant variables.

#### Phase 2: Variable Selection and Model Building
- Used the binned training dataset (`insurance_t_bin`).
- Treated all variables as categorical; replaced missing values with a "missing" category.
- Checked and addressed separation concerns.
- Built a main-effects logistic regression model using backward selection (𝛼 = 0.002).
- Ranked final model variables by p-value.
- Interpreted key odds ratios and explored notable findings.
- Investigated interaction effects using forward selection.

#### Phase 3: Model Assessment and Prediction
- Applied final model to binned training and validation datasets.
- Evaluated performance using:

  **Probability Metrics (Training):**
  - Concordance percentage
  - Discrimination slope

  **Classification Metrics (Training):**
  - ROC Curve (with visualizations)
  - K-S Statistic (for threshold determination,with visualizations)

  **Classification Metrics (Validation):**
  - Confusion matrix
  - Accuracy
  - Lift curve (with visualization)

- Reported insights on model performance and areas for improvement.

### Key Findings
- Strong predictors identified early, especially binary and ordinal variables.
- Odds ratios offered intuitive business insights on customer behavior.
- The model showed strong discriminative ability, separating buyers from non-buyers.
- ROC and lift charts confirmed predictive strength on validation data.

### Deliverables
- Code (commented R scripts)
- Executive-style report (PDF)
- Visualizations for key model evaluation metrics

### Notes
- Data used is anonymized and provided for academic purposes.
- Phase-wise structure ensures transparency and interpretability.
- Final output is designed to support strategic marketing decisions.

---

*Author: Ishi Gupta*  
*Last Updated: [March 27 2025]*

