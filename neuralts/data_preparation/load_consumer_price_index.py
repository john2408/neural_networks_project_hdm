# Loads file in the location 
# data/raw/statistisches_bundesamt/consumer_price_index.xlsx

import pandas as pd
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def load_consumer_price_index(file_path: str, sheet_name: str) -> pd.DataFrame:

    df = pd.read_excel(file_path, 
                    sheet_name=sheet_name)
    
    columns = ['Year','Month', 'consumer_price_index', "cpi_comment", "YoY_change", "YoY_change_comment", "MoM_change", "MoM_change_comment"]
    relevant_columns = ['Year','Month', 'consumer_price_index', "YoY_change", "MoM_change"]

    # skip first 2 rows 
    skiprows_head = 4
    skiprows_tail = 98
    df = df.iloc[skiprows_head:skiprows_tail].reset_index(drop=True)

    # Rename columns
    df.columns = columns

    # Keep only non-comment columns 
    df = df[relevant_columns].copy()
    
    # For coolumn "Year" Fill empty value with previous available value
    df['Year'].ffill(inplace=True)

    # Cast Columny Year and Monnt(str) to date as end of month
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str)) + pd.offsets.MonthEnd(0)

    # Cast numeric columns to float
    numeric_columns = ['consumer_price_index', "YoY_change"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    # Recalculate MoM_change based on consumer_price_index, since column contains NaN values
    df = df.sort_values(by=['Date']).reset_index(drop=True)
    df['MoM_change'] = np.round(df['consumer_price_index'].pct_change().bfill(),4)

    # Overwrite Month column with month extracted from Date
    df['Month'] = df['Date'].dt.month
    
    return df

if __name__ == "__main__":
    
    input_path = os.path.join(os.getcwd(), "data", "raw", "statistisches_bundesamt", "consumer_price_index.xlsx")
    output_path = os.path.join(os.getcwd(), "data", "processed", "historical_consumer_price_index.parquet")
    sheet_name = "61111-0002"

    df = load_consumer_price_index(file_path=input_path, sheet_name=sheet_name)
    df.to_parquet(output_path, engine='fastparquet', index=False)
    print(df.head())