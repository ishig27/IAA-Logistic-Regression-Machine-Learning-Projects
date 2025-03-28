# Banking Insurance Product - Machine Learning and Logistic Regression Analysis IAA Projects

This repository contains two predictive modeling projects, demonstrating statistical modeling and machine learning skills applied to a real-world dataset in the insurance domain. The goal is to predict customer behavior, specifically the likelihood of purchasing an insurance product, using various predictive methods.

## Project 1 - Logistic Regression
- **Objective:** Explore and model the relationship between various customer attributes and their likelihood of purchasing an insurance product.
- **Techniques Used:**
  - Logistic Regression modeling
  - Feature selection and ROC/AUC analysis
- **Outcome:** Identification of key factors influencing customer purchasing behavior.

## Project 2 - Machine Learning
- **Objective:** Enhance prediction accuracy by leveraging advanced machine learning methods.
- **Models Implemented:**
  - **GAM & MARS**
    - Non-linear models using Generalized Additive Models (GAM) and Multivariate Adaptive Regression Splines (MARS).
    - Performed cross-validation for robust model validation and identified influential predictors.
    - ROC curve and discrimination analysis.
  - **Random Forest:**
    - Initial model training, hyperparameter tuning, and variable importance.
    - Variable selection via permutation importance tests.
    - ROC curve and discrimination analysis.
  - **XGBoost:**
    - Cross-validation and hyperparameter tuning.
    - Feature selection through importance analysis.
    - ROC curve analysis and evaluation metrics.
  - **Final Model: EBM**
    - Initial model training, hyperparameter tuning, and variable importance.
        - Global and local interpretability.
        - Accumulated Local Effects (ALE) plots to understand feature influence.
    - Insights into specific variables such as Account Age, including targeted business recommendations.
- **Outcome:** Determined best-performing model with detailed performance metrics.


## Tools and Libraries Used
- **Python:** Pandas, NumPy, Scikit-learn, XGBoost, InterpretML, skexplain, Matplotlib
- **R:** tidyverse, randomForest, xgboost, caret, earth, mgcv

## Key Deliverables
- Comprehensive ROC/AUC performance evaluation.
- Feature importance and model interpretability plots.
- Actionable insights and recommendations based on model findings.


