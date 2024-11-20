import os
import zipfile
import requests
from tqdm import tqdm

# URLs and folder names
data = {
    "DKITCHEN_16k": "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip",
    "DWASHING_16k": "https://zenodo.org/records/1227121/files/DWASHING_16k.zip",
    "NFIELD_16k": "https://zenodo.org/records/1227121/files/NFIELD_16k.zip",
    "NPARK_16k": "https://zenodo.org/records/1227121/files/NPARK_16k.zip",
    "NRIVER_16k": "https://zenodo.org/records/1227121/files/NRIVER_16k.zip",
    "OHALLWAY_16k": "https://zenodo.org/records/1227121/files/OHALLWAY_16k.zip",
    "OMEETING_16k": "https://zenodo.org/records/1227121/files/OMEETING_16k.zip",
    "OOFFICE_16k": "https://zenodo.org/records/1227121/files/OOFFICE_16k.zip",
    "PCAFETER_16k": "https://zenodo.org/records/1227121/files/PCAFETER_16k.zip",
    "PRESTO_16k": "https://zenodo.org/records/1227121/files/PRESTO_16k.zip",
    "PSTATION_16k": "https://zenodo.org/records/1227121/files/PSTATION_16k.zip",
    "SPSQUARE_16k": "https://zenodo.org/records/1227121/files/SPSQUARE_16k.zip",
    "STRAFFIC_16k": "https://zenodo.org/records/1227121/files/STRAFFIC_16k.zip",
    "TBUS_16k": "https://zenodo.org/records/1227121/files/TBUS_16k.zip",
    "TCAR_16k": "https://zenodo.org/records/1227121/files/TCAR_16k.zip",
    "TMETRO_16k": "https://zenodo.org/records/1227121/files/TMETRO_16k.zip",
}

output_dir = "DEMND-Corpus"
os.makedirs(output_dir, exist_ok=True)

def download_file(url, output_path):

    response = requests.get(url, stream = True)
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file, tqdm(
        desc = f"{os.path.basename(output_path)}",
        total = total_size,
        unit = "B",
        unit_scale = True,
        unit_divisor = 1024,
    ) as bar:
        for data in response.iter_content(chunk_size = 1024):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_to):

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        zip_ref.extractall(extract_to)

for folder, url in data.items():

    folder_path = os.path.join(output_dir, folder)
    os.makedirs(folder_path, exist_ok = True)

    zip_file_path = os.path.join(output_dir, f"{folder}.zip")
    download_file(url, zip_file_path)

    print(f"extracting {zip_file_path} to {folder_path}")
    extract_zip(zip_file_path, folder_path)

    os.remove(zip_file_path)

print("all ready")

