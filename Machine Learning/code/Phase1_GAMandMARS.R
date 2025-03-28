# Machine Learning – Phase 1: MARS and GAM (R)
# Description: Builds and evaluates MARS and GAM classification models

# Load Required Libraries
library(tidyverse)
library(DescTools)
library(earth)
library(ROCit)
library(mgcv)

# Load Data
train <- read_csv("~/GitHub/IAA-Data-Science-projects/Machine Learning/data/insurance_t.csv", show_col_types = FALSE)
valid <- read_csv("~/GitHub/IAA-Data-Science-projects/Machine Learning/data/insurance_v.csv", show_col_types = FALSE)

# Count Unique levels for variables 
for (i in 1:ncol(train)) {
  print(colnames(train)[i])
  print(length(unique(train[, i][[1]])))
}

# Convert variables with <=10 unique values to factors
train <- train %>%
  mutate(across(c("DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "INV", 
                  "MM", "MMCRED", "CC", "CCPURC", "SDB", "INAREA"), as.factor))

valid <- valid %>%
  mutate(across(c("DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "INV", 
                  "MM", "MMCRED", "CC", "CCPURC", "SDB", "INAREA"), as.factor))

# Drop INS for imputation
train_imp <- select(train, -"INS")
valid_imp <- select(valid, -"INS")

# Impute missing values (median for numeric, mode for factor)
train_imp <- train_imp %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)),
         across(where(is.factor), ~ ifelse(is.na(.), as.character(Mode(., na.rm = TRUE)), as.character(.)))) %>%
  mutate(across(where(is.character), as.factor))
valid_imp <- valid_imp %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)),
         across(where(is.factor), ~ ifelse(is.na(.), as.character(Mode(., na.rm = TRUE)), as.character(.)))) %>%
  mutate(across(where(is.character), as.factor))

# INS back into data
train_imp <- cbind(train_imp, INS = train$INS)
valid_imp <- cbind(valid_imp, INS = valid$INS)


# ---------------------------
# MARS Model
# ---------------------------
mars <- earth(INS ~ ., data = train_imp, 
              glm = list(family = binomial(link = "logit")))
mars_cv <- earth(INS ~ ., data = train_imp, 
                 glm = list(family = binomial(link = "logit")), trace = .5, 
                 nfold = 5, pmethod = "cv")

# Variable Importance
print(evimp(mars))
print(evimp(mars_cv))

# Predict and Evaluate MARS Models (Training)
train_imp$mars_phat <- predict(mars, type = "response")
train_imp$mars_cv_phat <- predict(mars_cv, type = "response")

# ROC Curves
train_imp$mars_phat <- predict(mars, type = "response")[, 1]
mars_roc <- rocit(train_imp$mars_phat, train_imp$INS)
plot(mars_roc)
plot(mars_roc)$optimal
summary(mars_roc)

# Predict and Classify on Validation
valid_imp$INS_mars_pred <- predict(mars, newdata = valid_imp, type = "response")
valid_imp$INS_mars_cv_pred <- predict(mars_cv, newdata = valid_imp, type = "response")

# Use optimal thresholds (from training ROC)
valid_imp <- valid_imp %>%
  mutate(INS_mars_class = ifelse(INS_mars_pred > 0.30, 1, 0),
         INS_mars_cv_class = ifelse(INS_mars_cv_pred > 0.32, 1, 0))

# Accuracy
acc_mars <- mean(valid_imp$INS_mars_class == valid_imp$INS)
acc_mars_cv <- mean(valid_imp$INS_mars_cv_class == valid_imp$INS)

# ---------------------------
# GAM Model
# ---------------------------
gam <- gam(INS ~ s(ACCTAGE) + s(DDABAL) + s(DEP) + s(DEPAMT) + s(CHECKS) + 
             s(NSFAMT) + s(PHONE) + s(TELLER) + s(SAVBAL) + s(ATMAMT) + 
             s(POS) + s(POSAMT) + s(CDBAL) + s(IRABAL) + s(INVBAL) + s(MMBAL) + 
             s(CCBAL) + s(INCOME) + s(LORES) + s(HMVAL) + s(AGE) + s(CRSCORE) + 
             DDA + DIRDEP + NSF + SAV + ATM + CD + IRA + INV + MM + MMCRED + 
             CC + CCPURC + SDB + INAREA + BRANCH, 
           data = train_imp, family = binomial(), select = TRUE, method = "REML")

# Refined GAM Model
gam2 <- gam(INS ~ s(ACCTAGE) + s(DDABAL) + s(CHECKS) + s(TELLER) + s(SAVBAL) + 
              s(ATMAMT) + s(CDBAL) + s(CCBAL) + DDA + CD + IRA + INV + MM + CC + BRANCH,
            data = train_imp, family = binomial(), select = TRUE, method = "REML")

# ROC Curves for GAM
train_imp$gam_phat <- predict(gam, type = "response")
train_imp$gam2_phat <- predict(gam2, type = "response")

gam_roc <- rocit(train_imp$gam_phat, train_imp$INS)
gam2_roc <- rocit(train_imp$gam2_phat, train_imp$INS)
plot(gam_roc)
plot(gam2_roc)

# Predict and Classify GAM on Validation
gam_cutoff <- 0.31
valid_imp$INS_gam_pred <- predict(gam, newdata = valid_imp, type = "response")
valid_imp$INS_gam_class <- ifelse(valid_imp$INS_gam_pred > gam_cutoff, 1, 0)

# Accuracy
acc_gam <- mean(valid_imp$INS_gam_class == valid_imp$INS)

# Print Accuracies
cat("Validation Accuracy (MARS):", round(acc_mars, 3), "\n")
cat("Validation Accuracy (MARS-CV):", round(acc_mars_cv, 3), "\n")
cat("Validation Accuracy (GAM):", round(acc_gam, 3), "\n")
