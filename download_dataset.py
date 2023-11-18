import requests
import os
import zipfile

if __name__ == '__main__':
    # !pip install dicom2nifti
    # !pip install nilearn
    print(" - - - downloading started - - - ")

    url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download"
    response = requests.get(url)
    if response.status_code == 200:
        open("ACDC_temp.zip", "wb").write(response.content)
        print("File downloaded successfully.")
        zipfile.ZipFile("ACDC_temp.zip", 'r').extractall(os.getcwd())
        print("File unzipped successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
    os.remove("ACDC_temp.zip")
    
    print(" - - - downloading finished - - - ")