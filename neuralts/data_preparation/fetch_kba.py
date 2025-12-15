import os
import warnings

import requests

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    storage_path = os.path.join(os.getcwd(), "data/raw/kba/")
    os.makedirs(storage_path, exist_ok=True)

    base_url = "https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/"
    report_type = "fz10"

    year_months = [
        # Transformation Approach 1 - Sheet Name FZ 10.1
        "2025_10",
        "2025_09",
        "2025_08",
        "2025_07",
        "2025_06",
        "2025_05",
        "2025_04",
        "2025_03",
        "2025_02",
        "2025_01",
        "2024_12",
        "2024_11",
        "2024_10",
        "2024_09",
        "2024_08",
        "2024_07",
        "2024_06",
        "2024_05",
        "2024_04",
        "2024_03",
        "2024_02",
        "2024_01",
        # Transformation Approach 2 - Sheet Name FZ10.1
        "2023_12",
        "2023_11",
        "2023_10",
        "2023_09",
        "2023_08",
        "2023_07",
        "2023_06",
        "2023_05",
        "2023_04",
        "2023_03",
        "2023_02",
        "2023_01",
        "2022_12",
        "2022_11",
        "2022_10",
        "2022_09",
        "2022_08",
        "2022_07",
        "2022_06",
        "2022_05",
        "2022_04",
        "2022_03",
        "2022_02",
        "2022_01",
        "2021_12",
        "2021_11",
        "2021_10",
        "2021_09",
        "2021_08",
        "2021_07",
        "2021_06",
        "2021_05",
        "2021_04",
        "2021_03",
        "2021_02",
        "2021_01",
        # Different URL https://<...>FZ10/fz10_2020_12_xlsx.xlsx?__blob=publicationFile&v=2
        "2020_12",
        "2020_11",
        "2020_10",
        "2020_09",
        "2020_08",
        "2020_07",
        "2020_06",
        "2020_05",
        "2020_04",
        "2020_03",
        "2020_02",
        "2020_01",
        "2019_12",
        "2019_11",
        "2019_10",
        "2019_09",
        "2019_08",
        "2019_07",
        "2019_06",
        "2019_05",
        "2019_04",
        "2019_03",
        "2019_02",
        "2019_01",
        "2018_12",
        # Different URL https:/<...>z10_2018_11_xls.xls?__blob=publicationFile&v=2
        "2018_11",
        "2018_10",
        "2018_09",
        "2018_08",
        "2018_07",
        "2018_06",
        "2018_05",
        "2018_04",
        "2018_03",
        "2018_02",
        "2018_01",
    ]

    # Download Excel File
    for year_month in year_months:
        print("Downloading data for", year_month)

        # Determine URL and file extension based on year/month
        if year_month.startswith("2018"):
            month = year_month.split("_")[1]
            if month != "12":
                url = f"{base_url}{report_type}_{year_month}_xls.xls?__blob=publicationFile&v=2"
                file_path = os.path.join(storage_path, f"{report_type}_{year_month}.xls")
            else:
                url = f"{base_url}{report_type}_{year_month}_xlsx.xlsx?__blob=publicationFile&v=2"
                file_path = os.path.join(storage_path, f"{report_type}_{year_month}.xlsx")
        elif year_month.startswith("2020") or year_month.startswith("2019"):
            url = f"{base_url}{report_type}_{year_month}_xlsx.xlsx?__blob=publicationFile&v=2"
            file_path = os.path.join(storage_path, f"{report_type}_{year_month}.xlsx")
        else:
            url = f"{base_url}{report_type}_{year_month}.xlsx?__blob=publicationFile&v=2"
            file_path = os.path.join(storage_path, f"{report_type}_{year_month}.xlsx")

        # Check if file already exists to avoid re-downloading
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            continue

        response = requests.get(url)

        # Check if download was successful
        if response.status_code == 200:
            # Save the file locally
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"File downloaded and saved to {file_path}")
        else:
            print(f"Failed to download {year_month}. Status code: {response.status_code}")
