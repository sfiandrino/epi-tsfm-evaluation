import pandas as pd
from sodapy import Socrata


def remove_countries_with_skipped_last_data_update(df):
    """
    Remove countries that have skipped last data update
    """
    max_date = df.truth_date.max()
    location_max_dates = df.groupby("location", as_index=False).truth_date.max()
    for _, row in location_max_dates.iterrows():
        if row.truth_date != max_date:
            df = df[df.location != row.location]
    return df


def reindex_data(df, freq="W-SAT", interpol=True, method='linear'):
    """
    Reindex the data to include all dates in the range
    """
    all_dates = pd.date_range(df["truth_date"].min(), df["truth_date"].max(), freq=freq)
    all_countries = df["location"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_dates, all_countries],
        names=["truth_date", "location"]
    )
    df = (df.set_index(["truth_date", "location"])
            .reindex(full_index)          
            .reset_index())
    if interpol:
      # interpolate along each location's time series (rows must be ordered by location, then date)
      df = df.sort_values(by=["location", "truth_date"])
      df['value'] = df['value'].interpolate(
        method=method, limit_direction='both'
      )
    #df["target"].fillna(method="ffill", inplace=True)
    df["target"].ffill(inplace=True)
    return df


def import_data_US(target, full_date_range=True, dropna=False, interpol=True, method='linear', freq="W-SAT", skip_countries=True):
    """
    Import data from the US CDC
    Args:
        target: str, the target to import (totalconfflunewadm, totalconfc19newadm)
        full_date_range: bool, whether to include all dates in the range (i.e., fill in missing dates with NaN)
        dropna: bool, whether to drop rows with missing values
        interpol: bool, whether to interpolate the data
        method: str, the method to use for interpolation (linear, nearest, etc.)
        freq: str, the frequency of the data (W-SAT)
        skip_countries: bool, whether to skip countries with skipped last data update
    Returns:
        pd.DataFrame, the imported data
    """

    target_name_switch = {"totalconfflunewadm": "wk inc flu hosp", "totalconfc19newadm": "wk inc covid hosp"}

    # fetch data from CDC
    client = Socrata("data.cdc.gov", None)
    results = client.get("mpgq-jmmr", limit=100000)
    df = pd.DataFrame.from_records(results)
    df = df[["weekendingdate", "jurisdiction", target]]
    df['value'] = df[target]
    df['truth_date'] = pd.to_datetime(df['weekendingdate'])

    # change 'USA' to 'US'
    df['jurisdiction'] = df['jurisdiction'].apply(lambda x: 'US' if x == 'USA' else x)

    # add location information
    locations = pd.read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/auxiliary-data/locations.csv")
    df = df.merge(locations[["location", "abbreviation"]], left_on = ['jurisdiction'], right_on=['abbreviation']) 
    df["target"] = target_name_switch[target]
    df = df[['truth_date', 'location', "target", "value"]]

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["truth_date"] = pd.to_datetime(df["truth_date"])
    df = df.sort_values(["location", "truth_date"])

    # remove countries that have skipped last data update
    if skip_countries:
        df = remove_countries_with_skipped_last_data_update(df)

    # reindex to include all dates in the range
    if full_date_range:
        df = reindex_data(df, freq=freq, interpol=interpol, method=method)

    if dropna:
        df = df.dropna(subset=["value"])
    
    return df
