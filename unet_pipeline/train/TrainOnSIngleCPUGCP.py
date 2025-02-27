# python TrainOnSIngleGCP.py
# experiments/albunet_valid/train_config_part0.yaml --gcp_bucket
# "img-classif-training-dataset"
import argparse
import tempfile
import logging
import shutil
import subprocess

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import albumentations as albu
import torch

import importlib
import functools
from tqdm import tqdm
import os
from pathlib import Path

from unet_pipeline.utils.Pneumadataset import PneumothoraxDataset, PneumoSampler
from unet_pipeline.train.Learning import Learning
from utils.helpers import load_yaml, init_seed, init_logger
# from Evaluation import apply_deep_thresholds, search_deep_thresholds,
# dice_round_fn, search_thresholds
from google.cloud import storage


def list_files_in_bucket(bucket_name):
    """Lists all files in the given GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    return [blob.name for blob in blobs]

def check_local_directory_exists(local_data_dir, remote_data_dir):
    """Check if the remote data directory already exists in the local path."""
    local_remote_data_dir = local_data_dir 
    if local_remote_data_dir.exists() and any(local_remote_data_dir.iterdir()):
        print(f"Local directory {local_remote_data_dir} already exists.")
        return True
    else:
        print(f"Local directory {local_remote_data_dir} does not exist.")
        return False

def check_remote_directory_exists(bucket_name, remote_data_dir):
    """Check if the remote data directory exists in the GCP bucket."""
    remote_path = f"gs://{bucket_name}/{remote_data_dir}"
    try:
        # Run gsutil to check if the remote directory exists
        result = subprocess.run(['gsutil', 'ls', remote_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stdout:
            print(f"Remote directory {remote_path} exists.")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking remote directory {remote_path}: {e}")
        return False
    return False

def copy_folder_from_gcp(bucket_name, remote_data_dir, local_data_dir):
    """Copy an entire folder from GCP bucket to local directory using gsutil."""
    # Check if the local directory contains files
    
    # Check if the remote_data_dir already exists locally
    if check_local_directory_exists(local_data_dir, remote_data_dir):
        print(f"Skipping copy: The directory {remote_data_dir} already exists locally.")
        return  # Skip the copy if the directory exists

    # Check if the remote directory exists in the GCP bucket
    if not check_remote_directory_exists(bucket_name, remote_data_dir):
        print(f"Remote directory {remote_data_dir} does not exist in the bucket.")
        return  # Skip the copy if the remote directory doesn't exist
    
    # Ensure the local directory exists
    local_data_dir.mkdir(parents=True, exist_ok=True)

    # Define the remote and local paths
    remote_path = f"gs://{bucket_name}/{remote_data_dir}"
    local_path = str(local_data_dir)  # Convert Path object to string for gsutil

    # Run gsutil command to copy the folder recursively
    try:
        # Use subprocess to run the gsutil command
        subprocess.run(['gsutil', '-m', 'cp', '-r', remote_path + '/', local_path], check=True)
        print(f"Successfully copied {remote_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying {remote_path} to {local_path}: {e}")

def argparser():
    parser = argparse.ArgumentParser(description='Pneumatorax pipeline')
    parser.add_argument('train_cfg', type=str, help='train config path')
    parser.add_argument('--gcp_bucket', type=str,
                        required=True, help='GCP bucket name')
    return parser.parse_args()


def train_fold(
        train_config, experiment_folder, pipeline_name, log_dir, fold_id,
        train_dataloader, valid_dataloader, binarizer_fn, eval_fn):

    fold_logger = init_logger(log_dir, 'train_fold_{}.log'.format(fold_id))

    best_checkpoint_folder = Path(
        experiment_folder, train_config['CHECKPOINTS']['BEST_FOLDER'])
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config['CHECKPOINTS']['FULL_FOLDER'],
        'fold{}'.format(fold_id)
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config['CHECKPOINTS']['TOPK']

    calculation_name = '{}_fold{}'.format(pipeline_name, fold_id)

    device = train_config['DEVICE']

    module = importlib.import_module(train_config['MODEL']['PY'])
    model_class = getattr(module, train_config['MODEL']['CLASS'])
    model = model_class(**train_config['MODEL']['ARGS'])

    pretrained_model_config = train_config['MODEL'].get('PRETRAINED', False)
    if pretrained_model_config:
        loaded_pipeline_name = pretrained_model_config['PIPELINE_NAME']
        pretrained_model_path = Path(
            pretrained_model_config['PIPELINE_PATH'],
            pretrained_model_config['CHECKPOINTS_FOLDER'],
            '{}_fold{}.pth'.format(loaded_pipeline_name, fold_id)
        )
        if pretrained_model_path.is_file():
            model.load_state_dict(torch.load(pretrained_model_path))
            fold_logger.info(
                'load model from {}'.format(pretrained_model_path))

    if len(train_config['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)

    module = importlib.import_module(train_config['CRITERION']['PY'])
    loss_class = getattr(module, train_config['CRITERION']['CLASS'])
    loss_fn = loss_class(**train_config['CRITERION']['ARGS'])

    optimizer_class = getattr(torch.optim, train_config['OPTIMIZER']['CLASS'])
    optimizer = optimizer_class(
        model.parameters(), **train_config['OPTIMIZER']['ARGS'])
    scheduler_class = getattr(torch.optim.lr_scheduler,
                              train_config['SCHEDULER']['CLASS'])
    scheduler = scheduler_class(optimizer, **train_config['SCHEDULER']['ARGS'])

    n_epoches = train_config['EPOCHES']
    grad_clip = train_config['GRADIENT_CLIPPING']
    grad_accum = train_config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['EARLY_STOPPING']
    validation_frequency = train_config.get('VALIDATION_FREQUENCY', 1)

    freeze_model = train_config['MODEL']['FREEZE']

    Learning(
        optimizer,
        binarizer_fn,
        loss_fn,
        eval_fn,
        device,
        n_epoches,
        scheduler,
        freeze_model,
        grad_clip,
        grad_accum,
        early_stopping,
        validation_frequency,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        fold_logger
    ).run_train(model, train_dataloader, valid_dataloader)


# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    args = argparser()
    config_folder = Path(args.train_cfg.strip("/"))
    experiment_folder = config_folder.parents[0]

    # Load the configuration file
    train_config = load_yaml(config_folder)

    # List files in the GCP bucket
    bucket_name = args.gcp_bucket
    files = list_files_in_bucket(bucket_name)

    # Resolve the local data directory
    local_data_dir = Path(train_config['DATA_DIRECTORY']).resolve()
    # Remote data directory in GCP (provided by you)
    remote_data_dir = "dataset/dataset1024"  # You mentioned your dataset is under this folder

    # GCP bucket name (you specified this earlier)
    bucket_name = "img-classif-training-dataset"  # GCP bucket name

    # Copy the entire folder from remote to local
    copy_folder_from_gcp(bucket_name, remote_data_dir, local_data_dir)

    # Update the data directory in the config to use the local path
    train_config['DATA_DIRECTORY'] = str(local_data_dir)

    # Set up logging
    log_dir = Path(experiment_folder, train_config['LOGGER_DIR'])
    log_dir.mkdir(exist_ok=True, parents=True)
    main_logger = init_logger(log_dir, 'train_main.log')

    # Set random seed
    init_seed(train_config['SEED'])
    main_logger.info(train_config)

    # Handle GPU device settings
    if "DEVICE_LIST" in train_config:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            map(str, train_config["DEVICE_LIST"]))

    pipeline_name = train_config['PIPELINE_NAME']
    dataset_folder = train_config['DATA_DIRECTORY']

    # Validate and load transformations
    if 'TRAIN_TRANSFORMS' in train_config and 'VALID_TRANSFORMS' in train_config:
        train_transform = albu.load(train_config['TRAIN_TRANSFORMS'])
        valid_transform = albu.load(train_config['VALID_TRANSFORMS'])
    else:
        raise KeyError(
            "Missing TRAIN_TRANSFORMS or VALID_TRANSFORMS in train_config")

    non_empty_mask_proba = train_config.get('NON_EMPTY_MASK_PROBA', 0)
    use_sampler = train_config['USE_SAMPLER']

    folds_distr_path = train_config['FOLD']['FILE']
    num_workers = train_config['WORKERS']
    batch_size = train_config['BATCH_SIZE']
    n_folds = train_config['FOLD']['NUMBER']

    # Ensure correct handling of fold IDs
    usefolds = list(map(str, train_config['FOLD']['USEFOLDS']))

    # Load binarizer and evaluation function
    binarizer_module = importlib.import_module(
        train_config['MASK_BINARIZER']['PY'])
    binarizer_class = getattr(
        binarizer_module,
        train_config['MASK_BINARIZER']['CLASS'])
    binarizer_fn = binarizer_class(**train_config['MASK_BINARIZER']['ARGS'])

    eval_module = importlib.import_module(
        train_config['EVALUATION_METRIC']['PY'])
    eval_fn = getattr(eval_module, train_config['EVALUATION_METRIC']['CLASS'])
    eval_fn = functools.partial(
        eval_fn, **train_config['EVALUATION_METRIC']['ARGS'])

    for fold_id in usefolds:
        main_logger.info('Start training of {} fold....'.format(fold_id))

        # Train dataset and sampler
        train_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='train',
            transform=train_transform, fold_index=fold_id,
            folds_distr_path=folds_distr_path,
        )
        train_sampler = PneumoSampler(
            folds_distr_path, fold_id, non_empty_mask_proba)

        if use_sampler:
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=batch_size,
                num_workers=num_workers, sampler=train_sampler
            )
        else:
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=True
            )

        # Validation dataset and dataloader
        valid_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='val',
            transform=valid_transform, fold_index=str(fold_id),
            folds_distr_path=folds_distr_path,
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset, batch_size=batch_size,
            num_workers=num_workers, shuffle=False
        )

        # Start training
        train_fold(
            train_config, experiment_folder, pipeline_name, log_dir, fold_id,
            train_dataloader, valid_dataloader,
            binarizer_fn, eval_fn
        )

    # Upload results to GCP bucket
    gcp_handler.upload_folder(log_dir, f"results/{pipeline_name}")


if __name__ == "__main__":
    main()
