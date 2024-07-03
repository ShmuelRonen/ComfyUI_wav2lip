import os
import requests
import hashlib

# Funzione per calcolare il checksum SHA-256 di un file
def calculate_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file_from_sharepoint(file_name, file_info):
    file_url = file_info["link"]
    save_path = os.path.join(file_info["destination_folder"], file_name)
    # Controlla se il file esiste già
    if os.path.exists(save_path):
        # Verifica il checksum SHA-256
        computed_checksum = calculate_sha256(save_path)
        if computed_checksum == file_info["sha256_checksum"]:
            print(f"{file_name} OK")
            return
        else:
            print(f"{file_name} exists but has an incorrect SHA-256 checksum ({computed_checksum}). Initiating download for verification...")
    else:
        print(f"File {file_name} not found. Initiating download...")

    # Make GET request with allow_redirect
    res = requests.get(file_url, allow_redirects=True)

    if res.status_code == 200:
        # Get redirect url & cookies for using in next request
        new_url = res.url
        cookies = res.cookies.get_dict()
        for r in res.history:
            cookies.update(r.cookies.get_dict())

        # Do some magic on redirect url
        new_url = new_url.replace("onedrive.aspx", "download.aspx").replace("?id=", "?SourceUrl=")

        # Make new redirect request
        response = requests.get(new_url, cookies=cookies)

        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            # Verify SHA-256 checksum
            computed_checksum = calculate_sha256(save_path)
            if computed_checksum == file_info["sha256_checksum"]:
                print("File downloaded successfully!")
            else:
                print("Checksum verification failed! File corrupted or link invalid?")
                
        else:
            print("Failed to download the file.")
            print(response.status_code)
    else:
        print("Failed to get the redirect URL.")
        print(res.status_code)

# Funzione per scaricare un file e verificare il checksum SHA-256
def download_file_with_sha256(file_name, file_info):
    destination_folder = file_info["destination_folder"]
    destination_path = os.path.join(destination_folder, file_name)

    # Controlla se il file esiste già
    if os.path.exists(destination_path):
        # Verifica il checksum SHA-256
        computed_checksum = calculate_sha256(destination_path)
        if computed_checksum == file_info["sha256_checksum"]:
            print(f"{file_name} OK")
            return
        else:
            print(f"{file_name} exists but has an incorrect SHA-256 checksum ({computed_checksum}). Initiating download for verification...")
    else:
        print(f"File {file_name} not found. Initiating download...")

    # Esegue il download del file
    response = requests.get(file_info["link"])

    # Verifica se la richiesta è andata a buon fine (status code 200)
    if response.status_code == 200:
        # Crea la cartella se non esiste
        os.makedirs(destination_folder, exist_ok=True)

        # Scrive il contenuto nel file
        with open(destination_path, 'wb') as file:
            file.write(response.content)

        # Verifica nuovamente il checksum SHA-256
        computed_checksum = calculate_sha256(destination_path)
        if computed_checksum == file_info["sha256_checksum"]:
            print(f"Download complete: {destination_path}")
        else:
            print("Checksum verification failed! File corrupted or link invalid?")
            
    else:
        print(f"Error during download. Status code: {response.status_code}")



# Dictionary with file information (link and name)
files_to_download = {
    "wav2lip_gan.pth":{
        "link": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW",
        "destination_folder": "checkpoints",
        "sha256_checksum": "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
    },
    "visual_quality_disc.pth": {
    "link": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo",
    "destination_folder": "checkpoints",
    "sha256_checksum": "b3f8f6f7e954af02f2ffe0f3ea11f3259af89bff6e70933001c7c6bc8c145d96"
    },
    "lipsync_expert.pth": {
    "link": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP",
    "destination_folder": "checkpoints",
    "sha256_checksum": "9b9936c721696446eeed353032cab242a8cf0eed8c46cde540366f6ae5493be5"
    },
    "s3fd-619a316812.pth": {
        "link": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "destination_folder": "face_detection/detection/sfd",
        "sha256_checksum": "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"

    }
}

for file_name, file_info in files_to_download.items():
    if "sharepoint" in file_info["link"]:
        download_file_from_sharepoint(file_name, file_info)
    else:
        download_file_with_sha256(file_name, file_info)

