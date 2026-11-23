library(scoringutils)
library(hubData)
library("dplyr")
library("optparse")
library(readr)
library(purrr)
library(stringr)
library(tidyr)
library(zoo)
library(arrow)  # per read_parquet()
# Run this script from the code/ folder of the season.
getwd()
# arguments 
option_list = list(
  make_option("--hub_name", type = "character", default = "FluSight", help = "FluSight", metavar = "character"),
  make_option("--hub_path", type = "character", default = "../", help = "Hub path", metavar = "character"),
  make_option("--truth_file_name", type = "character", default = "surveillance_weekly.csv", help = "Latest truth file name", metavar = "character"), 
  make_option("--baseline_name", type = "character", default = "FluSight-baseline", help = "Latest truth file name", metavar = "character") 
);   

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser)

# mapping hub -> target label
target_labels <- c(
  "FluSight" = "wk inc flu hosp",
  "CovidHub" = "wk inc covid hosp"
)

target_label <- target_labels[[opt$hub_name]]

# print warning if hub name not recognized
if (is.null(target_label)) {
  stop(paste0("Hub name not recognized: ", opt$hub_name))
}

truth_data <- data.frame()
truth_data <- read.csv(paste0(opt$hub_path, "target-data/", opt$hub_name, "/", opt$truth_file_name), header = TRUE,
                       colClasses = c(location = "character"))

# rename truth columns 
truth_data <- truth_data %>% 
  rename("target_end_date" = "truth_date",
         "observed" = "value")
truth_data <- truth_data[, !names(truth_data) %in% "year_week"]
truth_data$target_end_date <- as.Date(truth_data$target_end_date, format = "%Y-%m-%d")

truth_data$target <- ifelse(truth_data$target == "", target_label, truth_data$target)
target <- unique(truth_data$target)
print(target)
print(head(truth_data))

# connect to hub
model_output_path <- file.path("..", "model-output", opt$hub_name)

# List of directories which are the models 
models <- list.dirs(
  model_output_path,
  recursive = FALSE,
  full.names = TRUE
)

# --- File CSV ---
files_csv <- list.files(
  model_output_path,
  pattern = "\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)

# --- File Parquet ---
files_parquet <- list.files(
  model_output_path,
  pattern = "\\.parquet$",
  recursive = TRUE,
  full.names = TRUE
)
print(files_parquet)

# Nomi modello da entrambi gli insiemi di file
file_names_csv     <- basename(files_csv)
file_names_parquet <- basename(files_parquet)

model_names <- gsub(
  "^\\d{4}-\\d{2}-\\d{2}-|\\.csv$|\\.parquet$",
  "",
  c(file_names_csv, file_names_parquet)
)

models_unique <- sort(unique(model_names))
print(models_unique)
# Hub models evaluated: models eligible under the 75% rule plus the six TSFMs.
# Eligibility is computed by code/eligibility/compute_eligibility.py (see code/eligibility/README.md):
# a model is eligible if it covers >= 75% of the forecast instances (reference date x location x
# horizon 0-3, all 23 quantiles) covered by the hub ensemble in the evaluation window.
elig <- read.csv(file.path(opt$hub_path, "..", "code", "eligibility", "eligibility_75pct.csv"))
eligible_models <- elig$model[elig$season == "2025/26" & elig$hub == opt$hub_name & as.logical(elig$eligible)]
tsfm_models <- c("Chronos-2", "Lag-Llama", "Moirai-2", "TabPFN-TS", "TimeGPT", "TimesFM")
models_unique <- models_unique[models_unique %in% c(eligible_models, tsfm_models)]
print(models_unique)

# --- Lettura CSV ---
model_outputs_csv <- map_dfr(
  files_csv,
  ~ read_csv(
    .x,
    show_col_types = FALSE,
    col_types = cols(
      reference_date = col_character(),
      horizon = col_double(),
      location = col_character(),
      target = col_character(),
      target_end_date = col_character(),
      output_type_id = col_double(),
      value = col_double()
    )
  ) %>%
    mutate(
      model = basename(.x) |>
        str_remove("^\\d{4}-\\d{2}-\\d{2}-") |>
        str_remove("\\.csv$")
    )
)

# --- Lettura Parquet ---
model_outputs_parquet <- map_dfr(
  files_parquet,
  ~ {
    df <- read_parquet(.x)
    # Allinea i tipi di colonna a quelli del CSV per evitare conflitti nel bind_rows
    df <- df %>%
      mutate(
        reference_date  = as.character(reference_date),
        target_end_date = as.character(target_end_date),
        horizon         = as.double(horizon),
        output_type_id  = as.double(output_type_id),
        value           = as.double(value),
        model = basename(.x) |>
          str_remove("^\\d{4}-\\d{2}-\\d{2}-") |>
          str_remove("\\.parquet$")
      )
    df
  }
)

# --- Unione ---
model_outputs <- bind_rows(model_outputs_csv, model_outputs_parquet) %>%
  filter(model %in% models_unique)

# Evaluation window (TSFM convention: reference_date = last week of data). Rounds outside it
# (e.g. hub rounds before the first TSFM round) are not scored.
eval_start <- "2025-11-22"
eval_end   <- "2026-05-23"
model_outputs <- model_outputs %>%
  filter(as.character(as.Date(reference_date)) >= eval_start,
         as.character(as.Date(reference_date)) <= eval_end)
# Official hub rounds only: reference dates on which the hub ensemble was produced (same
# definition as the eligibility denominator). Drops off-schedule submissions, e.g. 2025-01-18.
ensemble_name <- sub("-baseline$", "-ensemble", opt$baseline_name)
official_rounds <- unique(as.character(as.Date(model_outputs$reference_date[model_outputs$model == ensemble_name])))
model_outputs <- model_outputs %>%
  filter(as.character(as.Date(reference_date)) %in% official_rounds)
print(sort(unique(as.character(as.Date(model_outputs$reference_date)))))

model_outputs <- model_outputs %>%
  filter(
    model %in% models_unique)

check_models_names <- sort(unique(model_outputs$model))
print(check_models_names)

model_outputs <- model_outputs %>%
  mutate(
    reference_date  = as.Date(reference_date),
    target_end_date = as.Date(target_end_date)
  )

# Filter out rows with median, pmf as output type and check as target only the one related to inc hosp
model_outputs <- model_outputs %>%
  filter(
    output_type != "median",
    output_type != "pmf",
    target == target_label
  )

# Join model output and truth data (left join)
full_data <- merge(model_outputs, truth_data, by = c("target_end_date", "location", "target"), all.x = TRUE)

# remove rows where true data is null
full_data <- full_data %>%
  filter(!is.na(observed))

# Casting quantiles to 3 digits
full_data <- full_data %>%
  mutate(
    output_type_id = round(output_type_id, digits = 3) 
  )

full_data_clean <- full_data %>%
  rename(
    quantile_level = output_type_id,  # rename output_type_id → quantile
    predicted    = value       # rename prediction → value
  ) %>%
  mutate(
    quantile_level = as.numeric(quantile_level)  # convert quantile in numeric
  )

# Remove horizon -1 
full_data_clean <- full_data_clean %>%
  filter(
    !(horizon == -1) & 
      (output_type == 'quantile')
  )


# ------------- Monotonicity violation correction  ------------
full_data_clean <- full_data_clean %>%
  filter(!(horizon == -1) & (output_type == 'quantile')) %>%
  arrange(model, location, target, reference_date, target_end_date,
          horizon, quantile_level) %>%
  group_by(model, location, target, reference_date, target_end_date, horizon) %>%
  mutate(
    predicted_fixed = {
      vals <- predicted
      n <- length(vals)
      
      # Step 1: mark ONLY violating quantiles as NA
      last_valid <- vals[1]
      for (i in 2:n) {
        if (!is.na(vals[i]) && vals[i] < last_valid) {
          vals[i] <- NA
        } else if (!is.na(vals[i])) {
          last_valid <- vals[i]
        }
      }
      
      # Step 2: interpolate only where NA
      if (any(is.na(vals))) {
        vals <- approx(
          x      = seq_along(vals)[!is.na(vals)],
          y      = vals[!is.na(vals)],
          xout   = seq_along(vals),
          method = "linear",
          rule   = 2
        )$y
      }
      
      vals
    }
  ) %>%
  ungroup()

# save corrections in a correction summary to check models with monotonicity issue
corrections_summary <- full_data_clean %>%
  filter(abs(predicted_fixed - predicted) > 1e-10) %>%
  mutate(correction = predicted_fixed - predicted) %>%
  select(model, location, target, reference_date, target_end_date,
         horizon, quantile_level, predicted_original = predicted,
         predicted_fixed, correction) %>%
  arrange(desc(abs(correction)))

# substitute predicted with predicted_fixed
full_data_clean <- full_data_clean %>%
  select(-predicted) %>%
  rename(predicted = predicted_fixed)

# Log transformation (clamp values to 0 before transforming)
full_data_clean_log <- full_data_clean %>%
  mutate(
    observed  = log(observed + 1),
    predicted = log(pmax(predicted, 0) + 1)  # treats negatives as 0
  )

# compute scoring metrics
forecast_quantile <- full_data_clean_log %>%
  as_forecast_quantile(
    forecast_unit = c(
      "reference_date",   
      "target",           
      "target_end_date",  
      "horizon",         
      "location",         
      "model"
    )
  )

scores <- forecast_quantile %>%
  score()

metrics <- intersect(
  c("wis", "ae_median"),
  names(scores)
)
unique(scores$model)
print(opt$baseline_name)

# find locations where baseline is present
locations_with_baseline <- scores %>%
  filter(model == opt$baseline_name) %>%
  pull(location) %>%
  unique()

# keep only those locations
scores_filtered <- scores %>%
  filter(location %in% locations_with_baseline)

results <- lapply(metrics, function(m) {
  get_pairwise_comparisons(
    scores   = scores_filtered,
    by='location',
    compare  = "model",
    metric   = m,
    baseline = opt$baseline_name
  )
})

names(results) <- c("wis", "ae_median")
names(results)

library(data.table)

results_wis <- copy(results[["wis"]])
results_ae  <- copy(results[["ae_median"]])

# rename columns
setnames(
  results_wis,
  old = c("mean_scores_ratio", "wis_relative_skill", "wis_scaled_relative_skill"),
  new = c("mean_scores_ratio_wis", "wis_relative_skill", "wis_scaled_relative_skill"),
  skip_absent = TRUE
)

setnames(
  results_ae,
  old = c("mean_scores_ratio", "ae_median_relative_skill", "ae_median_scaled_relative_skill"),
  new = c("mean_scores_ratio_ae", "ae_median_relative_skill", "ae_median_scaled_relative_skill"),
  skip_absent = TRUE
)

key_cols <- c("model", "compare_against", "location")

final_results <- merge(
  results_wis,
  results_ae,
  by = key_cols,
  all = TRUE
)
write.csv(final_results, file = paste0(opt$hub_path, "/model-evaluation/forecast_pairwise_comparison_scores/forecast_scores_pairwise_comparison_summary_log_", opt$hub_name, ".csv"),
          row.names = FALSE)
write.csv(corrections_summary, file = paste0(opt$hub_path, "/model-evaluation/correction-quantiles/correction_quantiles_log_", opt$hub_name, ".csv"),
          row.names = FALSE)
write.csv(scores, file = paste0(opt$hub_path, "/model-evaluation/forecast_scores/forecast_scores_wis_ae_coverage_log_", opt$hub_name, ".csv"), row.names = FALSE)