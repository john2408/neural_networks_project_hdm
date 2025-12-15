# Load file from dat/raw/statitisches_bundesamt/population.csv
import os

import pandas as pd


def load_population_income_data(input_path: str) -> pd.DataFrame:
    """Loads and processes the population income data from Statistisches Bundesamt CSV file."""

    # Load the CSV file
    df = pd.read_csv(input_path, sep=";", skiprows=6, decimal=",")

    # Skip first row
    skiprows_head = 1
    skiprows_tail = 194
    df = df.iloc[skiprows_head:skiprows_tail].reset_index(drop=True)

    df.rename(
        columns={
            "Unnamed: 0": "Year",
            "Unnamed: 1": "Gender",
            "Unnamed: 2": "Income_Category",
        },
        inplace=True,
    )

    # Forward fill values
    df["Year"] = df["Year"].ffill()
    df["Gender"] = df["Gender"].ffill()

    # Convert Year column to datetime end of year
    df["Year"] = df["Year"].astype(int)

    # drop all raws with Income_Category as "Not specified"
    df = df[~df["Income_Category"].str.contains("Not specified", na=False)].copy()

    # drop columns with "Unnamed"
    cols_to_drop = [col for col in df.columns if "Unnamed" in col]
    df = df.drop(columns=cols_to_drop)

    cols_rename = {
        "Population in primary residence households": "Population_in_primary_residence_households",
        "Persons in employment from primary residence hh.": "Persons_in_employment_from_primary_residence_households",
        "Unemployed persons from primary residence hh.": "Unemployed_persons_from_primary_residence_households",
        "Economically active population from prim.resid.hh.": "Economically_active_population_from_primary_residence_households",
        "Economically inactive population fr.prim.resid.hh.": "Economically_inactive_population_from_primary_residence_households",
    }
    df = df.rename(columns=cols_rename)

    # Create data category column
    df["Data_Category"] = df["Gender"] + " - " + df["Income_Category"]

    # drop Gerder and Income_Category columns
    df = df.drop(columns=["Gender", "Income_Category"])

    numeric_cols = list(cols_rename.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].str.replace("/", "0", regex=False))

    # Validate no missing values remain
    try:
        assert not df.isnull().any().any()
    except AssertionError:
        print("Warning: There are still missing values in the DataFrame for file ", input_path)
        print(df.isnull().sum())

    # Pivot the DataFrame to have Data_Category as columns
    df = df.pivot(index="Year", columns="Data_Category", values=numeric_cols).reset_index()

    # Flatten MultiIndex columns
    df.columns = ["_".join(col).strip() if col[1] else col[0] for col in df.columns.values]

    return df


if __name__ == "__main__":

    input_path = os.path.join(
        os.getcwd(), "data", "raw", "statistisches_bundesamt", "population.csv"
    )
    output_path = os.path.join(
        os.getcwd(), "data", "processed", "population_income_germany.parquet"
    )
    df = load_population_income_data(input_path=input_path)
    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(df.head())
