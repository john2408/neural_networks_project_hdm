# Load employment level data
# Data Source: https://www.bundesbank.de/dynamic/action/en/statistics/time-series-databases/time-series-databases/745582/745582?tsId=BBDL1.M.DE.N.EMP.EBA000.A0000.A00.D00.0.ABA.A&listId=www_siws_mb09_06b&dateSelect=2025

# Load employment level data from location: data/raw/bundesbank/employment_level_germany.csv

import os

import numpy as np
import pandas as pd


def load_employment_level_data(input_path: str) -> pd.DataFrame:
    """Loads and processes the employment level data from Bundesbank CSV file."""

    columns = ["Date", "Employment_Level", "Status"]

    # Load the CSV file
    df = pd.read_csv(input_path, sep=",", skiprows=10, names=columns)

    # Drop Column "Status"
    df = df.drop(columns=["Status"])

    # Convert Date column from format 'YYYY-MM' to datetime end of month
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m") + pd.offsets.MonthEnd(0)

    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # Calculate month-to-month change in employment level
    df["Employment_Level_Change"] = df["Employment_Level"].diff().bfill()

    # Calculate month-to-month relative change in employment level
    df["Employment_Level_Relative_Change"] = np.round(
        df["Employment_Level"].pct_change().bfill(), 5
    )

    # Calculate year-to-year change in employment level
    df["Employment_Level_YoY_Change"] = df["Employment_Level"].diff(12).bfill()

    # Calculate year-to-year relative change in employment level
    df["Employment_Level_YoY_Relative_Change"] = np.round(
        df["Employment_Level"].pct_change(12).bfill(), 5
    )

    # Calculate 3-month moving average of employment level
    df["Employment_Level_MA_3"] = np.round(
        df["Employment_Level"].rolling(window=3).mean().bfill(), 5
    )

    # Calculate 6-month moving average of employment level
    df["Employment_Level_MA_6"] = np.round(
        df["Employment_Level"].rolling(window=6).mean().bfill(), 5
    )

    return df


if __name__ == "__main__":

    input_path = os.path.join(
        os.getcwd(), "data", "raw", "bundesbank", "employment_level_germany.csv"
    )
    output_path = os.path.join(
        os.getcwd(), "data", "processed", "historical_employment_level_germany.parquet"
    )
    df = load_employment_level_data(input_path=input_path)
    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(df.head())
