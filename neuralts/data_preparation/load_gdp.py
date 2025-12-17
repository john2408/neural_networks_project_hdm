# Load GDP data for Germany
# Data Source: https://www.bundesbank.de/dynamic/action/en/statistics/time-series-databases/time-series-databases/745582/745582?listId=www_ssb_lr_bip&tsId=BBNZ1.Q.DE.N.H.0000.L&dateSelect=2025
# Load from: data/raw/bundesbank/germany_GDP.csv

import os

import pandas as pd


def load_gdp_data(input_path: str) -> pd.DataFrame:
    """Loads and processes the employment level data from Bundesbank CSV file."""

    columns = ["Date", "GDP", "Status"]

    # Load the CSV file
    df = pd.read_csv(input_path, sep=",", skiprows=10, names=columns)

    # Drop Column "Status"
    df = df.drop(columns=["Status"])

    # Convert Date column from format 'YYYY-QQ' to datetime end of month
    df["Date"] = (
        df["Date"]
        .str.replace("Q1", "03")
        .str.replace("Q2", "06")
        .str.replace("Q3", "09")
        .str.replace("Q4", "12")
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m") + pd.offsets.MonthEnd(0)

    # Fill in missing months
    df = df.set_index("Date").resample("M").ffill().reset_index()

    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    return df


if __name__ == "__main__":

    input_path = os.path.join(os.getcwd(), "data", "raw", "bundesbank", "germany_GDP.csv")
    output_path = os.path.join(os.getcwd(), "data", "processed", "historical_GDP_germany.parquet")
    df = load_gdp_data(input_path=input_path)
    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(df.head())
