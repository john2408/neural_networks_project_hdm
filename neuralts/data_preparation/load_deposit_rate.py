# Load deposit rate from
# location: data/raw/bundesbank/ECB_deposit_facility_rate.csv

import os

import numpy as np
import pandas as pd


def load_deposit_rate_data(input_path: str) -> pd.DataFrame:

    df = pd.read_csv(input_path, sep=",", skiprows=9, decimal=".")

    skiprows_tail = 322
    df = df.iloc[:skiprows_tail].reset_index(drop=True)

    df.columns = ["Date", "Deposit_Rate", "Status"]

    # Drop Column "Status"
    df = df.drop(columns=["Status"])

    # Convert Date column from format 'YYYY-MM' to datetime end of month
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m") + pd.offsets.MonthEnd(0)

    df["Deposit_Rate"] = pd.to_numeric(df["Deposit_Rate"])

    # Calculate month-to-month change in Deposit rate
    df["Deposit_Rate_Change"] = df["Deposit_Rate"].diff().bfill()

    # Calculate month-to-month relative change in Deposit rate
    df["Deposit_Rate_Relative_Change"] = np.round(df["Deposit_Rate"].pct_change().bfill(), 5)

    # If Relative Percentage change is infinite (previous value was 0), set it to NaN and then bfill
    df["Deposit_Rate_Relative_Change"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["Deposit_Rate_Relative_Change"] = df["Deposit_Rate_Relative_Change"].bfill()
    # Calculate year-to-year change in Deposit rate
    df["Deposit_Rate_YoY_Change"] = df["Deposit_Rate"].diff(12).bfill()

    # Calculate year-to-year relative change in Deposit rate
    df["Deposit_Rate_YoY_Relative_Change"] = np.round(df["Deposit_Rate"].pct_change(12).bfill(), 5)

    return df


if __name__ == "__main__":

    input_path = os.path.join(
        os.getcwd(), "data", "raw", "bundesbank", "ECB_deposit_facility_rate.csv"
    )
    output_path = os.path.join(os.getcwd(), "data", "processed", "historical_deposit_rate.parquet")
    df = load_deposit_rate_data(input_path=input_path)
    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(df.head())
