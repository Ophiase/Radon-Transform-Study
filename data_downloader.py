import os
import requests
import zipfile
from constants import REAL_DATA_DIR, DOWNLOAD_URLS

def download_real_ct_data():
    """DICOM data downloader that extracts ZIP contents"""
    print("Starting DICOM data download...")
    
    for url in DOWNLOAD_URLS:
        try:
            # Get case number from URL
            case_num = url.split("case")[1].split(".zip")[0]
            zip_path = os.path.join(REAL_DATA_DIR, f"case_{case_num}.zip")
            
            # Download ZIP
            print(f"Downloading case {case_num}...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save temporary ZIP
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            # Extract directly to real_data
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(REAL_DATA_DIR)
            
            # Cleanup ZIP file
            os.remove(zip_path)
            print(f"Case {case_num} extracted successfully")
            
        except Exception as e:
            print(f"Error processing case {case_num}: {str(e)}")

if __name__ == "__main__":
    os.makedirs(REAL_DATA_DIR, exist_ok=True)
    download_real_ct_data()
    print("All downloads completed!")