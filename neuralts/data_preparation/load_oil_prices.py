# Historical Oil Prices are available from 
# https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en

import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def load_oil_prices(file_path: str) -> pd.DataFrame:
    """Loads historical oil prices from the specified CSV file."""
    
    sheet_name = "Prices with taxes"
    cols_to_keep = ['Date', 'GR_price_with_tax_euro95', 'GR_price_with_tax_diesel', 'GR_price_with_tax_heGRing_oil', 'GR_price_with_tax_fuel_oil_1']
    df = pd.read_excel(file_path, sheet_name, header=0)

    # keep only first column and 'Germany' column
    german_cols = [col for col in df.columns if 'GR' in col]
    df = df[[df.columns[0]] + german_cols]
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Drop rows with missing Date
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # # skip first two rows and last row for the footnotes
    df = df.iloc[1:-2].reset_index(drop=True)
   
    # keep only relevant columns
    df = df[['Date'] + german_cols].copy()

    # fillfoward missing values for numeric columns, first sort by date
    df.sort_values(by='Date', inplace=True)
    numeric_columns = df.columns.difference(['Date'])
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(method='ffill')
    
    # Keep only relevant columns
    df = df[cols_to_keep].copy()

    # Resample to monthly frequency, taking the last available price in the month
    df.set_index('Date', inplace=True)
    df = df.resample('M').last().reset_index()

    return df

if __name__ == "__main__":

    storage_path = os.path.join(os.getcwd(), "data/raw/european_commission/")

    os.makedirs(storage_path, exist_ok=True)

    file_path = os.path.join(storage_path, "Weekly_Oil_Bulletin_Prices_History_maticni_4web.xlsx")

    # Load oil prices
    df_oil_prices = load_oil_prices(file_path)
    print(df_oil_prices.shape)

    target_dir = os.path.join(os.getcwd(), "data/processed/historical_oil_prices.parquet")
    df_oil_prices.to_parquet(target_dir, engine='fastparquet', index=False)