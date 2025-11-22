import pandas as pd
import requests
import os
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    storage_path = os.path.join(os.getcwd(), "data/raw/kba/")
    os.makedirs(storage_path, exist_ok=True)

    base_url = "https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/"
    report_type = "fz10"

    year_months = [
                    
                # Transformation Approach 1 - Sheet Name FZ 10.1
                "2025_10", "2025_09", "2025_08", "2025_07", "2025_06", 
                "2025_05", "2025_04", "2025_03", "2025_02", "2025_01",
                "2024_12", "2024_11", "2024_10", "2024_09", "2024_08", 
                "2024_07", "2024_06", "2024_05", "2024_04", "2024_03", 
                "2024_02", "2024_01", 

                # Transformation Approach 2 - Sheet Name FZ10.1
                "2023_12", "2023_11", "2023_10", "2023_09", "2023_08",
                "2023_07", "2023_06", "2023_05", "2023_04", "2023_03",
                "2023_02", "2023_01",
                
                ]

    # Download Excel File
    for year_month in year_months:
        file_path = os.path.join(storage_path, f"{report_type}_{year_month}.xlsx")
        
        # Check if file already exists to avoid re-downloading
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            continue
            
        print("Downloading data for", year_month)
        url = f"{base_url}{report_type}_{year_month}.xlsx?__blob=publicationFile&v=2"

        response = requests.get(url)

        # Save the file locally
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved to {file_path}")
