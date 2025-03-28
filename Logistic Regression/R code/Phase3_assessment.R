# Logistic Regression - Phase 3 Model Assessment (R Version)
# Author(s): Ishi Gupta, Alison Bean de Hernandez, Austin Minihan, Patrick Costa
# Date: 2024-09-20

# Load required libraries
library(tidyverse)
library(DescTools)
library(mgcv)
library(vcdExtra)
library(writexl)
library(car)
library(survival)
library(ROCit)
library(caret)

# Load training and validation data
ins_train <- read.csv("~/GitHub/IAA-Data-Science-projects/Logistic Regression/data/insurance_t_bin.csv")
ins_valid <- read.csv("~/GitHub/IAA-Data-Science-projects/Logistic Regression/data/insurance_v_bin.csv")

# --------------------------
# Step 1: Handle Missing Values in Categorical Variables
# --------------------------
binary.vec <- c("DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "ILS", "LOC", "INV",
                "MM", "MTG", "CC", "SDB", "HMOWN", "MOVED", "INAREA")
ordin.vec <- c("CCPURC", "CASHBK", "MMCRED")
nomin.vec <- c("BRANCH", "RES")
categ.vec <- c(binary.vec, ordin.vec, nomin.vec)

fix_categorical <- function(df, categ.vec) {
  df_fix <- data.frame(temp = rep("temp", nrow(df)))
  for (var in categ.vec) {
    this.col <- ifelse(is.na(df[[var]]), "M", df[[var]])
    df_fix[[var]] <- this.col
  }
  df_fix[, -1]  # drop placeholder column
}

ins_train_fix <- fix_categorical(ins_train, categ.vec)
cont.vec <- setdiff(colnames(ins_train), c(categ.vec, "INS"))
ins_train_fix <- cbind(ins_train_fix, ins_train[, cont.vec])

# Collapse sparse levels to address separation
ins_train_fix$CASHBK <- ifelse(ins_train_fix$CASHBK == 2, 1, ins_train_fix$CASHBK)
ins_train_fix$MMCRED <- ifelse(ins_train_fix$MMCRED == 5, 3, ins_train_fix$MMCRED)

# Add response and convert to factors
ins_train_fix$INS <- ins_train$INS
ins_train_fix[] <- lapply(ins_train_fix, factor)

# --------------------------
# Step 2: Fit Final Logistic Model from Phase 2
# --------------------------
final.glm <- glm(INS ~ DDA + NSF + IRA + INV + MTG + DDABAL_BIN + CC + CHECKS_BIN +
                   TELLER_BIN + SAVBAL_BIN + ATMAMT_BIN + CDBAL_BIN + ILSBAL_BIN +
                   MMBAL_BIN + DDA:IRA,
                 family = binomial(link = "logit"), data = ins_train_fix)

# --------------------------
# Step 3: ANOVA Summary
# --------------------------
anova.results <- car::Anova(final.glm, test = 'LR', type = 'III', singular.ok = TRUE)
fwd.signif.df <- data.frame(variable = rownames(anova.results),
                            p.val = anova.results$`Pr(>Chisq)`) %>%
  filter(variable != "(Intercept)") %>% arrange(p.val)

# --------------------------
# Step 4: Validation Set Cleanup
# --------------------------
ins_valid_fix <- fix_categorical(ins_valid, categ.vec)
cont.vec <- setdiff(colnames(ins_valid), c(categ.vec, "INS"))
ins_valid_fix <- cbind(ins_valid_fix, ins_valid[, cont.vec])

# Collapse sparse levels for valid set
ins_valid_fix$CASHBK <- ifelse(ins_valid_fix$CASHBK == 2, 1, ins_valid_fix$CASHBK)
ins_valid_fix$MMCRED <- ifelse(ins_valid_fix$MMCRED == 5, 3, ins_valid_fix$MMCRED)

# Add response and convert to factors
ins_valid_fix$INS <- ins_valid$INS
ins_valid_fix[] <- lapply(ins_valid_fix, factor)

# --------------------------
# Step 5: Training Set Predictions
# --------------------------
ins_train_fix$p_hat <- predict(final.glm, type = "response")
p1 <- ins_train_fix$p_hat[ins_train_fix$INS == 1]
p0 <- ins_train_fix$p_hat[ins_train_fix$INS == 0]
coef_discrim <- mean(p1) - mean(p0)

# Plot coefficient of discrimination
ggplot(ins_train_fix, aes(p_hat, fill = factor(INS))) +
  geom_density(alpha = 0.7) +
  scale_fill_grey() +
  labs(x = "Predicted Probability", fill = "Outcome",
       title = paste("Coefficient of Discrimination = ", round(coef_discrim, 3)))

# --------------------------
# Step 6: ROC and KS Metrics (Training)
# --------------------------
logit_roc <- rocit(ins_train_fix$p_hat, factor(ins_train_fix$INS))
plot(logit_roc)
ksplot(logit_roc)
ks_stat <- ksplot(logit_roc)$`KS stat`

# Concordance metric
concordance(final.glm)

# --------------------------
# Step 7: Validation Set Prediction and Confusion Matrix
# --------------------------
ins_valid_fix$Pred <- predict(final.glm, newdata = ins_valid_fix, type = "response")
ins_valid_fix$Pred_INS <- ifelse(ins_valid_fix$Pred >= 0.2970672, 1, 0)  # threshold from KS plot

confusionMatrix(data = factor(ins_valid_fix$Pred_INS),
                reference = factor(ins_valid_fix$INS))

# --------------------------
# Step 8: Lift Chart
# --------------------------
logit_roc_valid <- rocit(ins_valid_fix$Pred, factor(ins_valid_fix$INS))
gt <- gainstable(logit_roc_valid)

# Create lift chart
plot(gt, type = 1)

# Optional: Store gain table
gt.df <- data.frame(gt$Lift)
gt.df$Depth <- gainstable(logit_roc)$Depth * 100
gt.df$CLift <- gainstable(logit_roc)$CLift

