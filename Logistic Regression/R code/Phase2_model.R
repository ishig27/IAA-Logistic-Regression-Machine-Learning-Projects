# Logistic Regression - Phase 2 (R Version)

# Load required libraries
library(tidyverse)      # Data wrangling
library(DescTools)      # Odds ratios
library(mgcv)           # GAMs 
library(vcdExtra)       # CMH tests 
library(writexl)        # Export Excel files
library(car)            # ANOVA (type III)

# Load binned training dataset
ins_train <- read.csv("~/GitHub/IAA-Data-Science-projects/Logistic Regression/data/insurance_t_bin.csv")

# --------------------------
# Step 1: Handle Missing Values in Categorical Variables
# --------------------------
# Replace all NA values with 'M' (missing category)
ins_train <- ins_train %>%
  mutate_all(~ replace(., is.na(.), "M"))

# --------------------------
# Step 2: Check for Separation Issues
# --------------------------
for (var in setdiff(colnames(ins_train), "INS")) {
  comCheck <- table(ins_train[[var]], ins_train$INS)
  if (any(comCheck == 0)) {
    print(var)  # Variables with separation issues
  }
}

# Manually resolve known separation issues by collapsing levels
ins_train$CASHBK <- as.character(ins_train$CASHBK)
ins_train$CASHBK[which(ins_train$CASHBK > 0)] <- "1+"

ins_train$MMCRED <- as.character(ins_train$MMCRED)
ins_train$MMCRED[which(ins_train$MMCRED > 2)] <- "3+"

# --------------------------
# Step 3: Backward Selection Model
# --------------------------
ins_train[] <- lapply(ins_train, factor)  # Convert all to factors

full.model <- glm(INS ~ ., data = ins_train, family = binomial())
empty.model <- glm(INS ~ 1, data = ins_train, family = binomial())

back.model <- step(full.model,
                   scope = list(lower = empty.model, upper = full.model),
                   direction = 'backward',
                   k = qchisq(0.002, 1, lower.tail = FALSE))

# Examine coefficients from final backward model
back.model.coef <- summary(back.model)$coefficients
signif.df <- data.frame(variable = rownames(back.model.coef),
                        p.val = back.model.coef[,"Pr(>|z|)"])
signif.df <- filter(signif.df, variable != "(Intercept)") %>% arrange(p.val)

# --------------------------
# Step 4: Main Effects Model (Manually Specified)
# --------------------------
main_effects_model <- glm(INS ~ DDA + NSF + IRA + INV + MTG + CC + DDABAL_BIN + 
                            CHECKS_BIN + TELLER_BIN + SAVBAL_BIN + ATMAMT_BIN + 
                            CDBAL_BIN + ILSBAL_BIN + MMBAL_BIN,
                          data = ins_train, family = binomial())

# --------------------------
# Step 5: Forward Selection on Interactions
# --------------------------
interaction_formula <- INS ~ (DDA + NSF + IRA + INV + MTG + CC + DDABAL_BIN + 
                                CHECKS_BIN + TELLER_BIN + SAVBAL_BIN + ATMAMT_BIN + 
                                CDBAL_BIN + ILSBAL_BIN + MMBAL_BIN)^2

full.model1 <- glm(interaction_formula, data = ins_train, family = binomial())
empty.model1 <- main_effects_model

forward.model1 <- step(empty.model1,
                       scope = list(lower = empty.model1, upper = full.model1),
                       direction = 'forward',
                       k = qchisq(0.002, 1, lower.tail = FALSE))

# --------------------------
# Step 6: Extract Forward Selection Results
# --------------------------
forward.model.coef <- summary(forward.model1)$coefficients
signif.df.forward <- data.frame(variable = rownames(forward.model.coef),
                                p.val = forward.model.coef[,"Pr(>|z|)"])
signif.df.forward <- filter(signif.df.forward, variable != "(Intercept)") %>% arrange(p.val)

# --------------------------
# Step 7: Final Model (Main + Key Interaction)
# --------------------------
final_effects_model <- glm(INS ~ DDA + NSF + IRA + INV + MTG + CC + DDABAL_BIN + 
                             CHECKS_BIN + TELLER_BIN + SAVBAL_BIN + ATMAMT_BIN + 
                             CDBAL_BIN + ILSBAL_BIN + MMBAL_BIN + DDA:IRA,
                           data = ins_train, family = binomial())

# --------------------------
# Step 8: ANOVA Output for Final Model
# --------------------------
anova.results <- Anova(forward.model1, test = 'LR', type = 'III', singular.ok = TRUE)
final.df <- data.frame(variable = rownames(anova.results),
                       p.val = anova.results$`Pr(>Chisq)`)
final.df <- filter(final.df, variable != "(Intercept)") %>% arrange(p.val)

# --------------------------
# Step 9: Export Results to Excel
# --------------------------
write_xlsx(final.df, "final_significant_terms.xlsx")
write_xlsx(signif.df.forward, "forward_model_significant_terms.xlsx")
