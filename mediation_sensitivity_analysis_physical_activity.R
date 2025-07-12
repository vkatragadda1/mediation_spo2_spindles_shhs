# SpO2–Spindle–Dementia Mediation Analysis: Sensitivity (additional covariate: physical activity)
# Runs mediation models with bootstrapping, calculates C-index and time-dependent AUC, exports to Excel.

# Load required packages
library(survival)
library(mediation)
library(openxlsx)
library(timeROC)

# Define fallback operator: returns 'a' if it exists and is not empty; otherwise returns 'b'
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0) a else b

# Set seed for reproducibility
set.seed(1234)

# Load data. Replace name with CSV file. 
df <- read.csv("INSERT CSV FILE WITH MEDIATION DATAFRAME")

# Define exposure variables and control/treated values, based on dataset characteristics 
exposures <- c("avg_spo2_no_desat_NREM", "hb_desat")
control_vals <- c(98, 22.34821792677382)
treated_vals <- c(90, 85.23912950737275)

# Define mediators
mediators <- c("SP_DENS_all_C", "SP_CHIRP_all_C", "SP_R_PHASE_IF_all_C")

# Define outcome columns
outcome_time <- "days_psg_to_dementia"
outcome_event <- "dementia_event"

# Define covariate group
covariate_sets <- list(Group4 = c("age","sex", "bmi", "broader_smoking", "broader_alcohol", 
                                  "broader_education", "group1_med", "group2_med","broader_exercise"))

# Create workbook for results 
wb <- createWorkbook()


# Set covariates and initialize results dataframe 
stage <- names(covariate_sets)[1]
covars <- covariate_sets[[stage]]

results <- data.frame()
  
# Loop through exposures, set control and treated values accordingly 
for (i in seq_along(exposures)) {
  exposure_var <- exposures[i]
  control_val <- control_vals[i]
  treat_val <- treated_vals[i]
    
  # Loop through mediators 
  for (med in mediators) {
    cat("Running:", exposure_var, "+", med, "with", stage, "\n")
    
    covar_formula <- if (length(covars) > 0) paste("+", paste(covars, collapse = " + ")) else ""
    
    # Fit mediator model (linear regression)
    formula_m <- as.formula(paste(med, "~", exposure_var, covar_formula))
    model.m <- lm(formula_m, data = df, na.action = na.omit)
    
    # Fit Weibull survival model 
    formula_y <- as.formula(paste("Surv(", outcome_time, ",", outcome_event, ") ~", med, "+", exposure_var, covar_formula))
    model.y <- survreg(formula_y, data = df, dist = "weibull", na.action = na.omit, control = list(maxiter = 2000))
    
    # Run mediation with bootstrapping (1,000 iterations)
    med.out <- try(mediate(model.m, model.y,
                           treat = exposure_var,
                           mediator = med,
                           outcome = outcome_time,
                           control.value = control_val,
                           treat.value = treat_val,
                           boot = TRUE, sims = 1000), silent = TRUE)
    
    # Below block calculates and prints performance metrics 
    if (!inherits(med.out, "try-error")) {
      summary_med <- summary(med.out)
      
      # RMSE + SD calculations 
      mediator_data <- model.frame(model.m)
      actual_vals <- mediator_data[[med]]
      predicted_vals <- predict(model.m)
      rmse_val <- sqrt(mean((predicted_vals - actual_vals)^2))
      std_dev_y <- sd(actual_vals)
      
      # Creating survival object to calculate AUC and C Index values 
      df[[outcome_time]] <- as.numeric(as.character(df[[outcome_time]]))
      df[[outcome_event]] <- as.numeric(as.character(df[[outcome_event]]))
      surv_obj <- Surv(df[[outcome_time]], df[[outcome_event]])
      linpred <- predict(model.y, type = "lp", newdata = df)
      
      # C-index calculation 
      c_index <- NA
      if (length(linpred) == nrow(df)) {
        concordance_result <- concordance(surv_obj ~ linpred)
        c_index <- round(concordance_result$concordance, 4)
        se <- sqrt(concordance_result$var)
        ci_lower <- c_index - 1.96 * se
        ci_upper <- c_index + 1.96 * se
      }
      if (is.null(c_index) || length(c_index) != 1 || is.na(c_index)) {
        c_index <- NA_real_
      }
      
      # AUC at 10 & 15 years
      auc_10y <- auc_15y <- NA
      auc_10y_ci <- auc_15y_ci <- c(NA_real_, NA_real_)
      
      tryCatch({
        roc_obj <- timeROC(T = df[[outcome_time]],
                           delta = df[[outcome_event]],
                           marker = -linpred,
                           cause = 1,
                           times = c(10*365, 15*365),
                           iid = TRUE)
        
        auc_vals <- roc_obj$AUC
        se_auc <- roc_obj$inference$vect_sd_1
        
        auc_10y <- round(auc_vals[2], 4)
        auc_15y <- round(auc_vals[3], 4)
        
        auc_10y_ci <- round(auc_vals[2] + c(-1.96, 1.96) * se_auc[2], 4)
        auc_15y_ci <- round(auc_vals[3] + c(-1.96, 1.96) * se_auc[3], 4)
        
      }, error = function(e) {
        cat("AUC calculation failed\n")
      })
      
      
      # Empirical p-values (manual calculation if needed) 
      # Note package prints p-values ONLY if all bootstrap iterations run. 
      acme_samples <- med.out$d0.sims
      ade_samples <- med.out$z0.sims
      te_samples <- med.out$tau.sims
      
      acme_pval_emp <- if (sum(!is.na(acme_samples)) > 0) {
        2 * min(mean(acme_samples <= 0, na.rm = TRUE), mean(acme_samples >= 0, na.rm = TRUE))
      } else NA
      ade_pval_emp <- if (sum(!is.na(ade_samples)) > 0) {
        2 * min(mean(ade_samples <= 0, na.rm = TRUE), mean(ade_samples >= 0, na.rm = TRUE))
      } else NA
      te_pval_emp <- if (sum(!is.na(te_samples)) > 0) {
        2 * min(mean(te_samples <= 0, na.rm = TRUE), mean(te_samples >= 0, na.rm = TRUE))
      } else NA
      
      # Bootstrap success rate 
      acme_bootstrap_success <- sum(!is.na(med.out$d0.sims))
      ade_bootstrap_success <- sum(!is.na(med.out$z0.sims))
      
      # Confidence intervals and p-values
      acme_ci <- summary_med$d0.ci %||% c(NA_real_, NA_real_)
      ade_ci <- summary_med$z0.ci %||% c(NA_real_, NA_real_)
      total_effect_ci <- summary_med$tau.ci %||% c(NA_real_, NA_real_)
      
      # Add values into dataframe 
      results <- rbind(results, data.frame(
        Exposure = exposure_var,
        Mediator = med,
        Covariate_Set = stage,
        ACME = summary_med$d0 %||% NA_real_,
        ACME_Control = summary_med$d0 %||% NA_real_,
        ACME_Treated = summary_med$d1 %||% NA_real_,
        ACME_CI_Lower = acme_ci[1],
        ACME_CI_Upper = acme_ci[2],
        ACME_pval_empirical = formatC(acme_pval_emp, format = "e", digits = 5),
        ADE = summary_med$z0 %||% NA_real_,
        ADE_Control = summary_med$z0 %||% NA_real_,
        ADE_Treated = summary_med$z1 %||% NA_real_,
        ADE_CI_Lower = ade_ci[1],
        ADE_CI_Upper = ade_ci[2],
        ADE_pval_empirical = formatC(ade_pval_emp, format = "e", digits = 5),
        Total_Effect = summary_med$tau.coef %||% NA_real_,
        Total_Effect_CI_Lower = total_effect_ci[1],
        Total_Effect_CI_Upper = total_effect_ci[2],
        TE_pval_empirical = formatC(te_pval_emp, format = "e", digits = 5),
        Prop_Mediated = summary_med$n0 %||% NA_real_,
        Prop_Mediated_Control = summary_med$n0 %||% NA_real_,
        Prop_Mediated_Treated = summary_med$n1 %||% NA_real_,
        Control_Value = control_val,
        Treated_Value = treat_val,
        RMSE = round(rmse_val, 5),
        Std_Dev_Y = round(std_dev_y, 5),
        C_Index = c_index,
        C_Index_CI_Lower = ci_lower,
        C_Index_CI_Upper = ci_upper,
        AUC_10y = auc_10y,
        AUC_10y_CI_Lower = auc_10y_ci[1],
        AUC_10y_CI_Upper = auc_10y_ci[2],
        AUC_15y = auc_15y,
        AUC_15y_CI_Lower = auc_15y_ci[1],
        AUC_15y_CI_Upper = auc_15y_ci[2],
        N_Used = nrow(model.frame(formula_y, data = df, na.action = na.omit)),
        ACME_Bootstrap_Success = acme_bootstrap_success,
        ADE_Bootstrap_Success = ade_bootstrap_success,
        stringsAsFactors = FALSE
      ))
    } else {
      cat("Mediation failed:", exposure_var, "+", med, "in", stage, "\n")
    }
  }
}
  
# Add sheet for covariate group, add in results dataframe into sheet 
addWorksheet(wb, sheetName = stage)
writeData(wb, sheet = stage, x = results)

# Save workbook for analysis 
saveWorkbook(wb, "mediation_results.xlsx", overwrite = TRUE)
