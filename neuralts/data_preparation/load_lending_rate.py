# Load interest rates data
# Data Source: Bundesbank https://www.bundesbank.de/dynamic/action/en/statistics/time-series-databases/time-series-databases/759784/759784?listId=www_szista_mb01
# Data is stored at: data/raw/bundesbank/CB_marginal_lending_facility_rate.csv
import os

import numpy as np
import pandas as pd


def load_lending_rate_data(input_path: str) -> pd.DataFrame:

    df = pd.read_csv(input_path, sep=",", skiprows=9, decimal=".")

    skiprows_tail = 322
    df = df.iloc[:skiprows_tail].reset_index(drop=True)

    df.columns = ["Date", "Lending_Rate", "Status"]

    # Drop Column "Status"
    df = df.drop(columns=["Status"])

    # Convert Date column from format 'YYYY-MM' to datetime end of month
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m") + pd.offsets.MonthEnd(0)

    df["Lending_Rate"] = pd.to_numeric(df["Lending_Rate"])

    # Calculate month-to-month change in lending rate
    df["Lending_Rate_Change"] = df["Lending_Rate"].diff().bfill()

    # Calculate month-to-month relative change in lending rate
    df["Lending_Rate_Relative_Change"] = np.round(df["Lending_Rate"].pct_change().bfill(), 5)

    # Calculate year-to-year change in lending rate
    df["Lending_Rate_YoY_Change"] = df["Lending_Rate"].diff(12).bfill()

    # Calculate year-to-year relative change in lending rate
    df["Lending_Rate_YoY_Relative_Change"] = np.round(df["Lending_Rate"].pct_change(12).bfill(), 5)

    return df


if __name__ == "__main__":

    input_path = os.path.join(
        os.getcwd(), "data", "raw", "bundesbank", "CB_marginal_lending_facility_rate.csv"
    )
    output_path = os.path.join(os.getcwd(), "data", "processed", "historical_lending_rate.parquet")
    df = load_lending_rate_data(input_path=input_path)
    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(df.head())
