import gdown
import os

def download_from_gdrive(file_id, dest_path):    
    #download demo weights if not exist
    os.makedirs("./pretrained", exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)