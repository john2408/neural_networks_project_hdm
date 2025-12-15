import pandas as pd 
import numpy as np 
import os
from dataclasses import dataclass

@dataclass
class GoldDataFrameConfig:
    """Configuration for Gold DataFrame creation"""

    kba_data_path: str = "data/processed/historical_kba_data.parquet"

    include_employment_level: bool = True
    employment_level_path: str = "data/processed/historical_employment_level_germany.parquet"   
    employment_level_cols_to_drop = ['Month', 'Year']

    include_consumer_price_index: bool = True
    consumer_price_index_path: str = "data/processed/historical_consumer_price_index.parquet"
    consumer_price_index_cols_to_drop = ['Month', 'Year']

    include_historical_deposit_rates: bool = True
    historical_deposit_rates_path: str = "data/processed/historical_deposit_rate.parquet"

    include_employment_level_germany: bool = True
    employment_level_germany_path: str = "data/processed/historical_employment_level_germany.parquet"
    employment_level_cols_to_drop = ['Month', 'Year']

    include_gdp_germany: bool = True
    gdp_germany_path: str = "data/processed/historical_GDP_germany.parquet"
    gdp_germany_cols_to_drop = ['Month', 'Year']

    include_historical_lending_rate:  bool = True
    historical_lending_rate_path: str = "data/processed/historical_lending_rate.parquet"

    include_historical_oil_prices: bool = True
    historical_oil_prices_path: str = "data/processed/historical_oil_prices.parquet"

    include_population_income_germany: bool = False
    population_income_germany_path: str = "data/processed/historical_population_income_germany.parquet"


def validate_no_missing_values(df: pd.DataFrame, file_path: str):
    """Validate that there are no missing values in the DataFrame."""
    try:
        assert df.isnull().any().any() == False
    except AssertionError:
        print("Warning: There are still missing values in the DataFrame for file ", file_path)
        print(df.isnull().sum())

def create_gold_dataframe(config: GoldDataFrameConfig) -> pd.DataFrame:
    """add all features to KBA dataframe to create gold dataframe"""


    # First load KBA data
    kba_df = pd.read_parquet(config.kba_data_path, engine='fastparquet')

    # Limit data to < 31.10.2025, since most features are only available up to Sep 2025 
    kba_df = kba_df[kba_df['Date'] < '2025-10-31'].copy()

    columns_to_keep = ['Date', 'ts_key', 'Value']

    df_gold = kba_df[columns_to_keep].copy()

    if config.include_employment_level:
        employment_level_df = pd.read_parquet(config.employment_level_path, engine='fastparquet')
        employment_level_df = employment_level_df.drop(columns=config.employment_level_cols_to_drop)

        # Merge with gold dataframe
        df_gold = df_gold.merge(employment_level_df, on='Date', how='left')
        
        # Validate no missing values after merge
        validate_no_missing_values(df_gold, config.employment_level_path)

    if config.include_consumer_price_index:
        consumer_price_index_df = pd.read_parquet(config.consumer_price_index_path, engine='fastparquet')
        consumer_price_index_df = consumer_price_index_df.drop(columns=config.consumer_price_index_cols_to_drop)

        # Merge with gold dataframe
        df_gold = df_gold.merge(consumer_price_index_df, on='Date', how='left')

        # Validate no missing values after merge
        validate_no_missing_values(df_gold, config.consumer_price_index_path)

    if config.include_historical_deposit_rates:
        deposit_rates_df = pd.read_parquet(config.historical_deposit_rates_path, engine='fastparquet')
        df_gold = df_gold.merge(deposit_rates_df, on='Date', how='left')

        validate_no_missing_values(df_gold, config.historical_deposit_rates_path)

    if config.include_employment_level_germany:
        employment_level_germany_df = pd.read_parquet(config.employment_level_germany_path, engine='fastparquet')
        employment_level_germany_df = employment_level_germany_df.drop(columns=config.employment_level_cols_to_drop)
        df_gold = df_gold.merge(employment_level_germany_df, on='Date', how='left')

        validate_no_missing_values(df_gold, config.employment_level_germany_path)

    if config.include_gdp_germany:
        gdp_germany_df = pd.read_parquet(config.gdp_germany_path, engine='fastparquet')
        gdp_germany_df = gdp_germany_df.drop(columns=config.gdp_germany_cols_to_drop)
        df_gold = df_gold.merge(gdp_germany_df, on='Date', how='left')

        validate_no_missing_values(df_gold, config.gdp_germany_path)

    if config.include_historical_lending_rate:
        lending_rate_df = pd.read_parquet(config.historical_lending_rate_path, engine='fastparquet')
        df_gold = df_gold.merge(lending_rate_df, on='Date', how='left')

        validate_no_missing_values(df_gold, config.historical_lending_rate_path)

    if config.include_historical_oil_prices:
        oil_prices_df = pd.read_parquet(config.historical_oil_prices_path, engine='fastparquet')
        df_gold = df_gold.merge(oil_prices_df, on='Date', how='left')

        validate_no_missing_values(df_gold, config.historical_oil_prices_path)
    
    assert df_gold.shape[0] == kba_df.shape[0], "Row count mismatch after merging features. Please check the merge operations."

    return df_gold 


if __name__ == "__main__":

    config = GoldDataFrameConfig()

    df_gold = create_gold_dataframe(config)

    output_path = os.path.join(os.getcwd(), "data/processed", "monthly_registration_volume_gold.parquet")
    df_gold.to_parquet(output_path, engine='fastparquet', index=False)
    print(f"Gold DataFrame saved to {output_path}")