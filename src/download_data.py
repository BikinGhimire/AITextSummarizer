import os
import json

# This MUST be set BEFORE importing KaggleApi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
os.environ['KAGGLE_CONFIG_DIR'] = project_root

# Import Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Function to authenticate kaggle API explicitly
def authenticate_kaggle():
    kaggle_json_path = os.path.join(project_root, 'kaggle.json')
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"Could not find {kaggle_json_path}")

    with open(kaggle_json_path, 'r') as f:
        creds = json.load(f)

    api = KaggleApi()
    api._load_config(creds)
    print("Kaggle authenticated successfully.")
    return api

# Function to download and extract data
def download_dataset(api, dataset_name, download_path='../data'):
    full_download_path = os.path.abspath(os.path.join(current_dir, download_path))
    os.makedirs(full_download_path, exist_ok=True)
    print(f'Downloading {dataset_name} dataset...')
    api.dataset_download_files(dataset_name, path=full_download_path, unzip=True)
    print('Download complete.')

if __name__ == '__main__':
    dataset = 'gowrishankarp/newspaper-text-summarization-cnn-dailymail'
    api = authenticate_kaggle()
    download_dataset(api, dataset)