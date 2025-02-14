import ray
import argparse
import subprocess
from pathlib import Path
from google.cloud import storage
from Pneumadataset import PneumothoraxDataset
from Learning import Learning
from utils.helpers import load_yaml, init_logger

def list_files_in_bucket(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs]

def check_local_directory_exists(local_data_dir):
    if local_data_dir.exists() and any(local_data_dir.iterdir()):
        print(f"Local directory {local_data_dir} already exists.")
        return True
    else:
        print(f"Local directory {local_data_dir} does not exist.")
        return False

def check_remote_directory_exists(bucket_name, remote_data_dir):
    remote_path = f"gs://{bucket_name}/{remote_data_dir}"
    try:
        result = subprocess.run(['gsutil', 'ls', remote_path], check=True, stdout=subprocess.PIPE)
        return bool(result.stdout)
    except subprocess.CalledProcessError:
        return False

def copy_folder_from_gcp(bucket_name, remote_data_dir, local_data_dir):
    if check_local_directory_exists(local_data_dir):
        return
    if not check_remote_directory_exists(bucket_name, remote_data_dir):
        return
    local_data_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"gs://{bucket_name}/{remote_data_dir}"
    subprocess.run(['gsutil', '-m', 'cp', '-r', f"{remote_path}/", str(local_data_dir)], check=True)

@ray.remote
def load_dataset_part(data_folder, mode, transform, fold_index, folds_distr_path):
    return PneumothoraxDataset(data_folder, mode, transform, fold_index, folds_distr_path)

def argparser():
    parser = argparse.ArgumentParser(description='Pneumothorax pipeline')
    parser.add_argument('train_cfg', type=str, help='train config path')
    parser.add_argument('--gcp_bucket', type=str, required=True, help='GCP bucket name')
    return parser.parse_args()

def train_fold(train_config, experiment_folder, pipeline_name, log_dir, fold_id, train_dataloader, valid_dataloader):
    logger = init_logger(log_dir, f'train_fold_{fold_id}.log')
    model_class = getattr(importlib.import_module(train_config['MODEL']['PY']), train_config['MODEL']['CLASS'])
    model = model_class(**train_config['MODEL']['ARGS'])
    learning = Learning(model, train_dataloader, valid_dataloader, logger)
    learning.run()

@ray.remote(num_cpus=4)
def train_fold_remote(*args):
    train_fold(*args)

def main():
    ray.init(address="auto")
    args = argparser()
    config_folder = Path(args.train_cfg.strip("/"))
    experiment_folder = config_folder.parents[0]
    train_config = load_yaml(config_folder)
    bucket_name = args.gcp_bucket
    local_data_dir = Path(train_config['DATA_DIRECTORY']).resolve()
    copy_folder_from_gcp(bucket_name, "dataset/data", local_data_dir)
    
    usefolds = range(train_config['FOLDS'])
    train_datasets = [load_dataset_part.remote(local_data_dir, 'train', None, fold_id, None) for fold_id in usefolds]
    valid_datasets = [load_dataset_part.remote(local_data_dir, 'val', None, fold_id, None) for fold_id in usefolds]
    train_datasets, valid_datasets = ray.get(train_datasets), ray.get(valid_datasets)
    
    futures = [train_fold_remote.remote(train_config, experiment_folder, "pipeline", "logs", fold_id, train_datasets[fold_id], valid_datasets[fold_id]) for fold_id in usefolds]
    ray.get(futures)

if __name__ == "__main__":
    main()
