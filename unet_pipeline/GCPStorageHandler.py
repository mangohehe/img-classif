import os
from google.cloud import storage
from pathlib import Path

class GCPStorageHandler:
    def __init__(self, bucket_name, credentials_path=None):
        """
        Initialize the GCP Storage Handler.

        Args:
            bucket_name (str): Name of the GCP bucket.
            credentials_path (str, optional): Path to the GCP credentials JSON file.
                                             If None, uses default credentials.
        """
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def download_file(self, remote_path, local_path):
        """
        Download a file from the GCP bucket to a local path.

        Args:
            remote_path (str): Path to the file in the GCP bucket.
            local_path (str): Local path to save the downloaded file.
        """
        blob = self.bucket.blob(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {remote_path} to {local_path}")

    def upload_file(self, local_path, remote_path):
        """
        Upload a file from a local path to the GCP bucket.

        Args:
            local_path (str): Local path of the file to upload.
            remote_path (str): Path to save the file in the GCP bucket.
        """
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to {remote_path}")

    def download_folder(self, remote_folder, local_folder):
        """
        Download all files from a folder in the GCP bucket to a local folder.

        Args:
            remote_folder (str): Path to the folder in the GCP bucket.
            local_folder (str): Local folder to save the downloaded files.
        """
        blobs = self.bucket.list_blobs(prefix=remote_folder)
        for blob in blobs:
            local_path = os.path.join(local_folder, os.path.relpath(blob.name, remote_folder))
            self.download_file(blob.name, local_path)

    def upload_folder(self, local_folder, remote_folder):
        """
        Upload all files from a local folder to a folder in the GCP bucket.

        Args:
            local_folder (str): Local folder containing files to upload.
            remote_folder (str): Path to the folder in the GCP bucket.
        """
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = os.path.join(remote_folder, os.path.relpath(local_path, local_folder))
                self.upload_file(local_path, remote_path)