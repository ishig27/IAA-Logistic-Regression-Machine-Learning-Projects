# Logistic Regression - Phase 1 Analysis (R Version)

# Load required libraries
library(tidyverse)      # For data manipulation and plotting
library(DescTools)      # For odds ratio calculations
library(mgcv)           # For GAM models and linearity checks
library(vcdExtra)       # For CMH tests
library(writexl)        # For writing Excel outputs

# Load training dataset
ins_train <- read.csv("~/GitHub/IAA-Data-Science-projects/Logistic Regression/data/insurance_t.csv")

# Define variable groups by type
binary.vec <- c("DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "ILS", "LOC", 
                "INV", "MM", "MTG", "CC", "SDB", "HMOWN", "MOVED", "INAREA")
ordin.vec <- c("CCPURC", "CASHBK", "MMCRED")
nomin.vec <- c("BRANCH", "RES")
cont.vec  <- c("ACCTAGE", "DDABAL", "DEP", "DEPAMT", "CHECKS", "NSFAMT", "PHONE", 
               "TELLER", "SAVBAL", "ATMAMT", "POS", "POSAMT", "CDBAL", "IRABAL", 
               "LOCBAL", "INVBAL", "ILSBAL", "MMBAL", "MTGBAL", "CCBAL", "INCOME", 
               "LORES", "HMVAL", "AGE", "CRSCORE")

# --------------------------
# Binary Variable Significance (CMH Test)
# --------------------------
pvals.vec <- c()
new.names <- c()

for (var in binary.vec) {
  test_result <- CMHtest(table(ins_train$INS, ins_train[[var]]))$table[1,][3]
  pvals.vec <- c(pvals.vec, test_result)
  new.names <- c(new.names, var)
}

signif.df <- data.frame(variable = new.names, p.val = pvals.vec, var.type = "Binary")

# --------------------------
# Nominal & Ordinal Variables
# --------------------------
nom_pvals <- c(chisq.test(table(ins_train$INS, ins_train$BRANCH))$p.val,
               chisq.test(table(ins_train$INS, ins_train$RES))$p.val)

ord_pvals <- c(fisher.test(table(ins_train$INS, ins_train$CCPURC))$p.val,
               fisher.test(table(ins_train$INS, ins_train$CASHBK))$p.val,
               fisher.test(table(ins_train$INS, ins_train$MMCRED))$p.val)

signif.df <- rbind(signif.df,
                   data.frame(variable = nomin.vec, p.val = nom_pvals, var.type = "Nominal"),
                   data.frame(variable = ordin.vec, p.val = ord_pvals, var.type = "Ordinal"))

# --------------------------
# Continuous Variables (LRT on logistic regression)
# --------------------------
anovas.vec <- c()

for (var in cont.vec) {
  filtered <- ins_train %>% filter(!is.na(.data[[var]]))
  full_model <- glm(INS ~ filtered[[var]], data = filtered, family = binomial())
  reduced_model <- glm(INS ~ 1, data = filtered, family = binomial())
  anovas.vec <- c(anovas.vec, anova(full_model, reduced_model, test = "LRT")$`Pr(>Chi)`[2])
}

signif.df <- rbind(signif.df,
                   data.frame(variable = cont.vec, p.val = anovas.vec, var.type = "Continuous"))

# --------------------------
# Filter & Sort Significant Variables (alpha = 0.002)
# --------------------------
significant_vars <- signif.df %>% filter(p.val < 0.002) %>% arrange(p.val)

# --------------------------
# Odds Ratios for Binary Variables
# --------------------------
odds.df <- signif.df %>% filter(var.type == "Binary")
odds.vec <- c()

for (var in odds.df$variable) {
  odds.vec <- c(odds.vec, OddsRatio(table(ins_train$INS, ins_train[[var]])))
}
odds.df$odds.val <- odds.vec
odds.df <- odds.df %>% arrange(desc(odds.val))

# --------------------------
# Linearity Assumption for Continuous Variables (edf < 2 is approximately linear)
# --------------------------
edf.vec <- c()

for (var in significant_vars %>% filter(var.type == "Continuous") %>% pull(variable)) {
  filtered <- ins_train %>% filter(!is.na(.data[[var]]))
  formula_text <- paste0("INS ~ s(", var, ")")
  gam_model <- gam(as.formula(formula_text), data = filtered, family = binomial(), method = "REML")
  edf.vec <- c(edf.vec, summary(gam_model)$edf)
}

edf.df <- data.frame(variable = significant_vars %>% filter(var.type == "Continuous") %>% pull(variable),
                     edf = edf.vec)

# --------------------------
# Missing Values Summary
# --------------------------
missing_counts <- sapply(ins_train, function(x) sum(is.na(x)))
missing.df <- data.frame(variable = names(missing_counts), n.missing = missing_counts)

# --------------------------
# Final Exported Outputs (for report)
# --------------------------
write_xlsx(arrange(signif.df, p.val), 'all_var_table.xlsx')
write_xlsx(significant_vars, 'signif_var_table.xlsx')
write_xlsx(odds.df, 'odds_table.xlsx')

# --------------------------
# Visualization: Missing Data
# --------------------------
ggplot(data = filter(missing.df, n.missing > 0), aes(x = reorder(variable, -n.missing), y = n.missing)) +
  geom_bar(stat = "identity", fill = 'lightblue') +
  labs(x = "Variable Name", y = "Number of Missing Values", title = "Missing Values by Variable") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
