from dataclasses import dataclass
import os

import pandas as pd


@dataclass
class GoldDataFrameConfig:
    """Configuration for Gold DataFrame creation"""

    kba_data_path: str = "data/processed/historical_kba_data.parquet"

    include_employment_level: bool = True
    employment_level_path: str = "data/processed/historical_employment_level_germany.parquet"
    employment_level_cols_to_drop = ["Month", "Year"]

    include_consumer_price_index: bool = True
    consumer_price_index_path: str = "data/processed/historical_consumer_price_index.parquet"
    consumer_price_index_cols_to_drop = ["Month", "Year"]

    include_historical_deposit_rates: bool = True
    historical_deposit_rates_path: str = "data/processed/historical_deposit_rate.parquet"
    historical_deposit_rates_cols_to_drop = ["Deposit_Rate_YoY_Relative_Change"]

    include_gdp_germany: bool = True
    gdp_germany_path: str = "data/processed/historical_GDP_germany.parquet"
    gdp_germany_cols_to_drop = ["Month", "Year"]

    include_historical_lending_rate: bool = True
    historical_lending_rate_path: str = "data/processed/historical_lending_rate.parquet"

    include_historical_oil_prices: bool = True
    historical_oil_prices_path: str = "data/processed/historical_oil_prices.parquet"

    include_population_income_germany: bool = False
    population_income_germany_path: str = (
        "data/processed/historical_population_income_germany.parquet"
    )


def validate_no_missing_values(df: pd.DataFrame, file_path: str):
    """Validate that there are no missing values in the DataFrame."""
    try:
        assert not df.isnull().any().any()
    except AssertionError:
        print("Warning: There are still missing values in the DataFrame for file ", file_path)
        print(df.isnull().sum())


def apply_timeseries_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common time series cleaning steps to the DataFrame."""


    print("Initial shape before cleaning:", df.shape)
    print("Initial number of unique ts_keys:", df['ts_key'].nunique())

    # Apply some data Quality Checks for the timeseries     
    df["ts_key_size"] = df.groupby('ts_key')['ts_key'].transform('size')

    # Filter ts_keys with at least 12 entries
    df = df[df['ts_key_size'] >= 12].copy()

    # Do not include timeseries which have the last 12 months as zero values
    recent_12_months = df['Date'].max() - pd.DateOffset(months=12)
    recent_data = df[df['Date'] > recent_12_months]

    # Remove ts_keys with all zero values in the last 12 months
    zero_value_ts_keys = recent_data.groupby('ts_key')['Value'].sum()
    zero_value_ts_keys = zero_value_ts_keys[zero_value_ts_keys == 0].index
    df = df[~df['ts_key'].isin(zero_value_ts_keys)].copy()

    # keep only ts_key contain data until max date 
    max_date = df['Date'].max()
    ts_keys_with_max_date = df[df['Date'] == max_date]['ts_key'].unique()
    df = df[df['ts_key'].isin(ts_keys_with_max_date)].copy()

    # Limit data to < 31.10.2025, since most features are only available up to Sep 2025
    df = df[df["Date"] < "2025-10-31"].copy()

    # All negative values are set to zero
    df.loc[df["Value"] < 0, "Value"] = 0

    # For repetead ts_key and Date, take the mean value
    df = df.groupby(['ts_key', 'Date'], as_index=False)['Value'].mean()

    print("Final shape after cleaning:", df.shape)
    print("Final number of unique ts_keys:", df['ts_key'].nunique())

    return df


# For Multivariate Forecasting, all ts must have the same length
def pad_timeseries_to_same_length(df):
    """
    Pad all time series to have the same length by adding zero-padded rows
    at the beginning and end for series that start later or end earlier.
    
    Input DataFrame must have columns: Date, ts_key, Value
    This represents model classes that were not yet available at earlier dates,
    or were discontinued before the end date.
    """
    # Get global date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='ME')
    
    print(f"\nPadding time series to same length:")
    print(f"  Global date range: {min_date} to {max_date}")
    print(f"  Total timesteps: {len(all_dates)}")
    
    # Get all unique time series keys
    all_ts_keys = df['ts_key'].unique()
    
    # Create complete DataFrame with all dates for all time series
    padded_rows = {}
    
    padded_rows['Date'] = []
    padded_rows['ts_key'] = []
    padded_rows['Value'] = []

    for ts_key in all_ts_keys:
        ts_data = df[df['ts_key'] == ts_key].sort_values('Date')
        
        missing_dates_start = set(all_dates) - set(ts_data['Date'])

        padded_rows['Date'].extend(missing_dates_start)
        padded_rows['ts_key'].extend([ts_key] * len(missing_dates_start))
        padded_rows['Value'].extend([0.0] * len(missing_dates_start))
    

    # Combine original data with padded rows
    if padded_rows:
        df_padded = pd.concat([df, pd.DataFrame(padded_rows)], ignore_index=True)
        df_padded = df_padded.sort_values(['ts_key', 'Date']).reset_index(drop=True)
        print(f"  Padding added for {len(padded_rows['Date'])} missing timesteps across series")
    else:
        df_padded = df.copy()
        print(f"  No padding needed - all series span full date range")
    
    # Validate all time series now have same length
    ts_lengths = df_padded.groupby('ts_key').size()
    assert ts_lengths.nunique() == 1, f"Time series have different lengths:\n{ts_lengths.value_counts()}"
    print(f"  ✓ All {len(all_ts_keys)} time series now have {ts_lengths.iloc[0]} timesteps")
    
    return df_padded

def create_gold_dataframe(config: GoldDataFrameConfig, zero_padding: bool = False) -> pd.DataFrame:
    """add all features to KBA dataframe to create gold dataframe"""

    # First load KBA data
    kba_df = pd.read_parquet(config.kba_data_path, engine="pyarrow")
    columns_to_keep = ["Date", "ts_key", "Value"]

    df_gold = kba_df[columns_to_keep].copy()

    df_gold = apply_timeseries_cleaning(df_gold)

    if zero_padding:
        df_gold = pad_timeseries_to_same_length(df_gold)



    if config.include_employment_level:
        employment_level_df = pd.read_parquet(config.employment_level_path, engine="pyarrow")
        employment_level_df = employment_level_df.drop(
            columns=config.employment_level_cols_to_drop
        )

        # Merge with gold dataframe
        df_gold = df_gold.merge(employment_level_df, on="Date", how="left")

        # Validate no missing values after merge
        validate_no_missing_values(df_gold, config.employment_level_path)

    if config.include_consumer_price_index:
        consumer_price_index_df = pd.read_parquet(
            config.consumer_price_index_path, engine="pyarrow"
        )
        consumer_price_index_df = consumer_price_index_df.drop(
            columns=config.consumer_price_index_cols_to_drop
        )

        # Merge with gold dataframe
        df_gold = df_gold.merge(consumer_price_index_df, on="Date", how="left")

        # Validate no missing values after merge
        validate_no_missing_values(df_gold, config.consumer_price_index_path)

    if config.include_historical_deposit_rates:
        deposit_rates_df = pd.read_parquet(
            config.historical_deposit_rates_path, engine="pyarrow"
        )
        deposit_rates_df = deposit_rates_df.drop(
            columns=config.historical_deposit_rates_cols_to_drop
        )
        df_gold = df_gold.merge(deposit_rates_df, on="Date", how="left")

        validate_no_missing_values(df_gold, config.historical_deposit_rates_path)


    if config.include_gdp_germany:
        gdp_germany_df = pd.read_parquet(config.gdp_germany_path, engine="pyarrow")
        gdp_germany_df = gdp_germany_df.drop(columns=config.gdp_germany_cols_to_drop)
        df_gold = df_gold.merge(gdp_germany_df, on="Date", how="left")

        validate_no_missing_values(df_gold, config.gdp_germany_path)

    if config.include_historical_lending_rate:
        lending_rate_df = pd.read_parquet(
            config.historical_lending_rate_path, engine="pyarrow"
        )
        df_gold = df_gold.merge(lending_rate_df, on="Date", how="left")

        validate_no_missing_values(df_gold, config.historical_lending_rate_path)

    if config.include_historical_oil_prices:
        oil_prices_df = pd.read_parquet(config.historical_oil_prices_path, engine="pyarrow")
        df_gold = df_gold.merge(oil_prices_df, on="Date", how="left")

        validate_no_missing_values(df_gold, config.historical_oil_prices_path)



    # Fill infity values with interpolation of neughboring values
    df_gold.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df_gold.interpolate(method='linear', inplace=True)

    # Check if any column has infinite values
    if df_gold.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("DataFrame contains infinite values. Please check the data.")
    
    validate_no_missing_values(df_gold, "None")

    if not zero_padding:
        # Ensure row count matches original KBA data if no padding applied
        assert (
            df_gold.shape[0] == kba_df.shape[0]
            ), "Row count mismatch after merging features. Please check the merge operations."

    # Sort by ts_key and Date
    df_gold = df_gold.sort_values(by=["ts_key", "Date"]).reset_index(drop=True)

    return df_gold



if __name__ == "__main__":

    zero_padding = True  # Set to True to enable zero padding of time series

    config = GoldDataFrameConfig()

    df_gold = create_gold_dataframe(config, zero_padding=zero_padding)

    
    if zero_padding: 
        file_name = "monthly_registration_volume_gold_padding.parquet"
    else: 
        file_name = "monthly_registration_volume_gold.parquet"

    output_path = os.path.join(
        os.getcwd(), "data/gold", file_name
    )
    df_gold.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Gold DataFrame saved to {output_path}")
