library(scoringutils)
library(hubData)
library("dplyr")
library("optparse")
library(readr)
library(purrr)
library(stringr)
library(tidyr)
library(zoo)
library(MMWRweek)
library(epidatr)
# Run this script from the code/ folder of the season.
getwd()


forecast_dates <- seq(as.Date("2026-05-19"), as.Date("2026-06-15"), by=7) - 3 # wednesdays
all_dates = list()
i = 1
for (d in forecast_dates) {
  date = as.Date(d)
  print(date)
  #date = paste0(MMWRweek(date)$MMWRyear, sprintf("%02d", MMWRweek(date)$MMWRweek))
  date = paste0(MMWRweek(date)$MMWRyear, sprintf("%02d", MMWRweek(date)$MMWRweek))
  nation <- pub_covidcast(
    source = 'nhsn',
    signals = "confirmed_admissions_covid_ew_prelim",
    time_type = "week",
    # time_values = epirange("2020-10-01", "2025-04-15"),
    geo_type = "nation",
    as_of = date
  )
  nation$as_of <- as.Date(d)
  nation$geo_type <- nation$direction <- nation$lag <- nation$missing_sample_size <- nation$missing_stderr <- nation$missing_value <- nation$stderr <- nation$sample_size <- NULL
  state <- pub_covidcast(
    source = 'nhsn',
    signals = "confirmed_admissions_covid_ew_prelim",
    time_type = "week",
    # time_values = epirange("2020-10-01", "2025-04-15"),
    geo_type = "state",
    as_of = date
  )
  state$as_of <- as.Date(d)
  state$geo_type <- state$direction <- state$lag <- state$missing_sample_size <- state$missing_stderr <- state$missing_value <- state$stderr <- state$sample_size <- NULL
  
  df = rbind(nation,state)
  df$target_end_date <- df$time_value + 6
  all_dates[[i]] = df
  print(i/length(forecast_dates))
  i = i + 1
  write.csv(df, paste0("covid_hosp26_", as.Date(d), '.csv'))
}