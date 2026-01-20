import os


def download_data(data_path, drive_link="google_drive_link"):

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    folder_id = drive_link.split("/folders/")[1].split("?")[0]
    os.system(f"gdown --folder https://drive.google.com/drive/folders/{folder_id} -O {data_path}") # Downloads the data to data_path
    
