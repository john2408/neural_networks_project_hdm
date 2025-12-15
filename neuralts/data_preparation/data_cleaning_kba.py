import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def data_cleaning(file_path: str, skiprows: int = 7) -> pd.DataFrame:
    """Cleans the KBA Excel file and returns a cleaned DataFrame."""

    # Rename German columns to short English versions
    column_mapping = {
        "Insgesamt": "Total",
        "mit Dieselantrieb": "Diesel",
        # Variants of Hybridantrieb naming in different files
        "mit Hybridantrieb": "Hybrid",
        "mit Hybridantrieb (incl. \nPlug-in-Hybrid)": "Hybrid",
        "mit Hybridantrieb (incl. Plug-in-Hybrid)": "Hybrid",
        "mit Hybridantrieb\n(incl. Plug-in-Hybrid)": "Hybrid",
        "mit Hybridantrieb \n(incl. Plug-in-Hybrid)": "Hybrid",
        "Benzin-Hybridantrieb \n(incl. Plug-in-Hybrid)": "Hybrid_Petrol_All",
        "Diesel-Hybridantrieb \n(incl. Plug-in-Hybrid)": "Hybrid_Diesel_All",
        "Hybridantrieb \n(ohne Plug-in-Hybrid)": "Hybrid_NonPlugin",
        "Benzin-Hybridantrieb \n(ohne Plug-in-Hybrid)": "Hybrid_Petrol_NonPlugin",
        "Diesel-Hybridantrieb \n(ohne Plug-in-Hybrid)": "Hybrid_Diesel_NonPlugin",
        "Plug-in-Hybridantrieb": "Hybrid_Plugin",
        "Benzin-Plug-in-Hybridantrieb": "Hybrid_Petrol_Plugin",
        "Diesel-Plug-in-Hybridantrieb": "Hybrid_Diesel_Plugin",
        "mit Elektroantrieb (BEV)": "Electric_BEV",
        "mit Elektroantrieb": "Electric_BEV",
        "mit Allradantrieb": "All_Wheel_Drive",
        "Cabriolets": "Convertibles",
    }

    file_name = file_path.split("/")[-1]
    year, month = file_name.split("_")[1], file_name.split("_")[-1].split(".")[0]
    year_month_date = pd.to_datetime(f"{year}-{month}-01") + pd.offsets.MonthEnd(0)

    # Load the Excel file
    # Check existing sheet names to determine correct sheet name
    xls = pd.ExcelFile(file_path, engine="openpyxl" if file_path.endswith(".xlsx") else "xlrd")

    sheet_name = [
        sheet
        for sheet in xls.sheet_names
        if "FZ" in sheet or "Pkw-NZL Marken, Modellreihen" in sheet
    ]

    assert len(sheet_name) == 1, f"No sheet with 'FZ' found in file {file_path}"

    sheet_name = sheet_name[0]

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate where the dataframe starts
    for row_idx in range(5, 10):
        if any(isinstance(col, str) and "Insgesamt" in col for col in df.iloc[row_idx]):
            skiprows = row_idx
            break
    else:
        raise ValueError(f"Could not find correct header row in file {file_path}.")
    # Rename columns and set proper headers
    new_columns = [x if isinstance(x, str) else "Unnamed" for x in df.iloc[skiprows]]
    new_columns[1] = "OEM"
    new_columns[2] = "Model"
    df.columns = new_columns

    # Drop all rows before the header row
    df = df.iloc[skiprows + 1:].reset_index(drop=True)

    # Filter all values after value "NEUZULASSUNGEN INSGESAMT"
    index_of_last_row = df[df["OEM"].str.contains("NEUZULASSUNGEN INSGESAMT").fillna(False)].index[
        0
    ]
    df = df.iloc[1:index_of_last_row]

    # Select only columns with absolute numbers
    selected_columns = [x for x in df.columns if not ("Unnamed" in x)]
    df = df[selected_columns].copy()

    # Filter out rows where OEM is 'ZUSAMMEN'
    df = df[~df["OEM"].str.contains("ZUSAMMEN", na=False)]

    # Fill out missing values in 'OEM' column
    df["OEM"].ffill(inplace=True)

    # Fill out missing values and any string values with 0
    numeric_columns = df.columns.difference(["OEM", "Model"])
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Fill NanN in Column Model with "SONSTIGE"
    # Examles are OEMs like MAXUS and MORGAN which have no specific model names
    df["Model"].fillna("SONSTIGE", inplace=True)

    # IF OEM is Null, drop the rows
    df.dropna(subset=["OEM"], inplace=True)

    # Validate no missing values remain
    try:
        assert not df.isnull().any().any()
    except AssertionError:
        print("Warning: There are still missing values in the DataFrame for file ", file_path)
        print(df.isnull().sum())

    df.rename(columns=column_mapping, inplace=True)

    # Calculate 'Petrol' columns as Total - Diesel - Electric - Hybrid
    df["Petrol"] = df["Total"] - df["Diesel"] - df["Electric_BEV"] - df["Hybrid"]

    final_columns = [
        "OEM",
        "Model",
        "Total",
        "Petrol",
        "Diesel",
        "Electric_BEV",
        "Hybrid",
        "All_Wheel_Drive",
        "Convertibles",
    ]

    df = df[final_columns].copy()

    # Reshape DataFrame from wide to long format
    df = df.melt(id_vars=["OEM", "Model"], var_name="drive_type", value_name="Value")

    df["Date"] = year_month_date

    df["ts_key"] = df["OEM"] + "_" + df["Model"] + "_" + df["drive_type"]

    # sort values
    df.sort_values(by=["Date", "ts_key"], inplace=True)

    return df


if __name__ == "__main__":

    storage_path = os.path.join(os.getcwd(), "data/raw/kba/")

    # List all donwnloaded files
    available_files = [
        file
        for file in os.listdir(storage_path)
        if file.endswith(".xlsx") or file.endswith(".xls")
    ]
    print("Downloaded files:", available_files)

    dfs = []

    for file_name in available_files:
        # for file_name in ['fz10_2018_04.xls']:

        file_path = os.path.join(storage_path, file_name)

        df_cleaned = data_cleaning(file_path)

        dfs.append(df_cleaned)

        del df_cleaned

    # Join all dataframes
    df = pd.concat(dfs)

    print(df.shape)

    # Save cleaned data
    cleaned_data_path = os.path.join(os.getcwd(), "data/processed/historical_kba_data.parquet")
    df.to_parquet(cleaned_data_path)
