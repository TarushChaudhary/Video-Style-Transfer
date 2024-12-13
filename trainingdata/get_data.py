import zipfile
from torch.hub import download_url_to_file
import os


def get_mscoco_data():
    
    MS_COCO_2014_TRAIN_DATASET_PATH = r'http://images.cocodataset.org/zips/train2014.zip'  # ~13 GB after unzipping


    print(f'Downloading from {MS_COCO_2014_TRAIN_DATASET_PATH}')
    resource_tmp_path = 'mscoco_dataset.zip'
    download_url_to_file(MS_COCO_2014_TRAIN_DATASET_PATH, resource_tmp_path)

    print(f'Started unzipping...')
    with zipfile.ZipFile(resource_tmp_path) as zf:
        local_resource_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'mscoco')
        os.makedirs(local_resource_path, exist_ok=True)
        zf.extractall(path=local_resource_path)
    print(f'Unzipping to: {local_resource_path} finished.')

    os.remove(resource_tmp_path)
    print(f'Removing tmp file {resource_tmp_path}.')


    
def check_for_msco_data():
    local_resource_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'mscoco')
    if not os.path.exists(local_resource_path) or not os.listdir(local_resource_path):
        print(f'Directory {local_resource_path} does not exist or is empty. Downloading MS COCO data...')
        get_mscoco_data()

