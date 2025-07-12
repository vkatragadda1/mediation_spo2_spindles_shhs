# SpO2–Spindle–Dementia Five-Fold Internal Validation, Mediation Analysis
# Runs mediation models with bootstrapping, calculates C-index and time-dependent AUC, exports to Excel.
# Runs models across 5 folds, 4 covariate tiers, and all 6 mediation models 

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
df <- read.csv("INSERT CSV FILE WITH MEDIATION DATAFRAME.csv")

# Confirm each fold is treated as categorical from csv, rather than numerical 
df$fold <- as.factor(df$fold)
# Create a vector of folds to loop through in mediation 
folds <- levels(df$fold)


# Define exposure variables and control/treated values, based on dataset characteristics 
exposures <- c("avg_spo2_no_desat_NREM", "hb_desat")
control_vals <- c(98, 22.34821792677382)
treated_vals <- c(90, 85.23912950737275)

# Define mediators
mediators <- c("SP_DENS_all_C", "SP_CHIRP_all_C", "SP_R_PHASE_IF_all_C")

# Define outcome columns
outcome_time <- "days_psg_to_dementia"
outcome_event <- "dementia_event"

# Define covariate groups
covariate_sets <- list(
  Unadjusted = c(),
  Group1 = c("age", "sex", "bmi"),
  Group2 = c("age", "sex", "bmi", "smoking", "alcohol", "education"),
  Group3 = c("age", "sex", "bmi", "smoking", "alcohol", "education", "group1_med", "group2_med")
)

# Create workbook for results 
wb <- createWorkbook()


# Loop through mediation for each of the five folds 
for (fold_id in folds) {
  cat("Processing Fold:", fold_id, "\n")
  
  # Training = all folds except current; Validation = current fold
  train_df <- df[df$fold != fold_id, ]
  test_df  <- df[df$fold == fold_id, ]
  
  # Initialize results dataframe once per fold
  results <- data.frame()  
  
  # Loop through all 6 exposure-mediator pairs and through each covariate set 
  for (stage in names(covariate_sets)) {
    covars <- covariate_sets[[stage]]
    
    # Loop through exposures, set control and treated values accordingly 
    for (i in seq_along(exposures)) {
      exposure_var <- exposures[i]
      control_val <- control_vals[i]
      treat_val <- treated_vals[i]
      
      # Loop through mediators 
      for (med in mediators) {
        cat("Running:", exposure_var, "+", med, "with", stage, "\n")
        covar_formula <- if (length(covars) > 0) paste("+", paste(covars, collapse = " + ")) else ""
        
        # Fit mediator model (linear regression) on the training dataset 
        formula_m <- as.formula(paste(med, "~", exposure_var, covar_formula))
        model.m <- lm(formula_m, data = train_df, na.action = na.omit)
        
        # Fit Weibull survival model on the training dataset 
        formula_y <- as.formula(paste("Surv(", outcome_time, ",", outcome_event, ") ~", med, "+", exposure_var, covar_formula))
        model.y <- survreg(formula_y, data = train_df, dist = "weibull", na.action = na.omit, control = list(maxiter = 2000))
        
        # Run mediation with bootstrapping (1,000 iterations) on the training models 
        med.out <- try(mediate(model.m, model.y,
                               treat = exposure_var,
                               mediator = med,
                               outcome = outcome_time,
                               control.value = control_val,
                               treat.value = treat_val,
                               boot = TRUE, sims = 1000), silent = FALSE)
        
        # Below block calculates and prints performance metrics 
        if (!inherits(med.out, "try-error")) {
          summary_med <- summary(med.out)
          
          
          # RMSE + SD calculation on training data -- metrics to show how well the linear model fits data 
          mediator_data <- model.frame(model.m)
          actual_vals <- mediator_data[[med]]
          predicted_vals <- predict(model.m)
          rmse_val <- sqrt(mean((predicted_vals - actual_vals)^2))
          std_dev_y <- sd(actual_vals)
          
          # C-index and AUC on test set -- metrics to evaluate how the outcome model performs on "new data" (test set)
          # Creating survival object to calculate values on test data 
          test_df[[outcome_time]] <- as.numeric(as.character(test_df[[outcome_time]]))
          test_df[[outcome_event]] <- as.numeric(as.character(test_df[[outcome_event]]))
          surv_obj <- Surv(test_df[[outcome_time]], test_df[[outcome_event]])
          
          # Create linear model prediction based on test data 
          linpred <- predict(model.y, type = "lp", newdata = test_df)
          
          
          # C-index
          c_index <- NA
          if (length(linpred) == nrow(test_df)) {
            concordance_result <- concordance(surv_obj ~ linpred)
            c_index <- round(concordance_result$concordance, 4)
            se <- sqrt(concordance_result$var)
            ci_lower <- c_index - 1.96 * se
            ci_upper <- c_index + 1.96 * se
          }
          if (is.null(c_index) || length(c_index) != 1 || is.na(c_index)) {
            c_index <- NA_real_
          }
          
          # AUC at 10 and 15 years
          auc_10y <- NA
          auc_15y <- NA
          auc_10y_ci <- auc_15y_ci <- c(NA_real_, NA_real_)
          tryCatch({
            roc_obj <- timeROC(T = test_df[[outcome_time]],
                               delta = test_df[[outcome_event]],
                               marker = -linpred,
                               cause = 1,
                               times = c(10*365, 15*365),
                               iid = TRUE)
    
            auc_vals <- roc_obj$AUC
            se_auc <- roc_obj$inference$vect_sd_1
            
            auc_10y <- round(roc_obj$AUC[2], 4)
            auc_15y <- round(roc_obj$AUC[3], 4)
            
            auc_10y_ci <- round(auc_vals[2] + c(-1.96, 1.96) * se_auc[2], 4)
            auc_15y_ci <- round(auc_vals[3] + c(-1.96, 1.96) * se_auc[3], 4)
            
          }, error = function(e) {
            cat("AUC calculation failed\n")
          })
          
          # Empirical p-values
          acme_samples <- med.out$d0.sims
          ade_samples <- med.out$z0.sims
          acme_pval_emp <- if (sum(!is.na(acme_samples)) > 0) {
            2 * min(mean(acme_samples <= 0, na.rm = TRUE), mean(acme_samples >= 0, na.rm = TRUE))
          } else NA
          ade_pval_emp <- if (sum(!is.na(ade_samples)) > 0) {
            2 * min(mean(ade_samples <= 0, na.rm = TRUE), mean(ade_samples >= 0, na.rm = TRUE))
          } else NA
          
          # Extract additional CIs and p-value
          ade_ci <- summary_med$z0.ci %||% c(NA_real_, NA_real_)
          te_ci <- summary_med$tau.ci %||% c(NA_real_, NA_real_)
          te_pval <- summary_med$tau.p %||% NA_real_
          tau_samples <- med.out$tau.sims
          tau_pval_emp <- if (sum(!is.na(tau_samples)) > 0) {
            2 * min(mean(tau_samples <= 0, na.rm = TRUE), mean(tau_samples >= 0, na.rm = TRUE))
          } else NA
          
          # Bootstrap success
          acme_bootstrap_success <- sum(!is.na(med.out$d0.sims))
          ade_bootstrap_success <- sum(!is.na(med.out$z0.sims))
          
          # Confidence intervals and p-values
          acme_ci <- summary_med$d0.ci %||% c(NA_real_, NA_real_)
          ade_ci <- summary_med$z0.ci %||% c(NA_real_, NA_real_)
          total_effect_ci <- summary_med$tau.ci %||% c(NA_real_, NA_real_)
          
          # Add all values into dataframe to export 
          results <- rbind(results, data.frame(
            Exposure = exposure_var,
            Mediator = med,
            Covariate_Set = stage,
            Fold = fold_id,
            ACME = summary_med$d0 %||% NA_real_,
            ACME_Control = summary_med$d0 %||% NA_real_,
            ACME_Treated = summary_med$d1 %||% NA_real_,
            ACME_CI_Lower = acme_ci[1],
            ACME_CI_Upper = acme_ci[2],
            ADE = summary_med$z0 %||% NA_real_,
            ADE_Control = summary_med$z0 %||% NA_real_,
            ADE_Treated = summary_med$z1 %||% NA_real_,
            ADE_CI_Lower = ade_ci[1],
            ADE_CI_Upper = ade_ci[2],
            Total_Effect_CI_Lower = te_ci[1],
            Total_Effect_CI_Upper = te_ci[2],
            Total_Effect_pval = tau_pval_emp %||% NA_real_,
            Total_Effect = summary_med$tau.coef %||% NA_real_,
            Prop_Mediated = summary_med$n0 %||% NA_real_,
            Prop_Mediated_Control = summary_med$n0 %||% NA_real_,
            Prop_Mediated_Treated = summary_med$n1 %||% NA_real_,
            Control_Value = control_val,
            Treated_Value = treat_val,
            ACME_pval_empirical = formatC(acme_pval_emp, format = "e", digits = 5),
            ADE_pval_empirical = formatC(ade_pval_emp, format = "e", digits = 5),
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

  }
  # Add results from 4 covariate groups x 6 mediation models into corresponding fold's sheet 
  sheet_name <- paste0("Fold_", fold_id)
  addWorksheet(wb, sheetName = sheet_name)
  writeData(wb, sheet = sheet_name, x = results)
}

# Save final results workbook 
saveWorkbook(wb, "five_fold_validation_results.xlsx", overwrite = TRUE)
