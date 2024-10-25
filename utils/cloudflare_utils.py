import os
import boto3
import botocore
import threading
from tqdm import tqdm
from loguru import logger


class ProgressPercentage:
    def __init__(self, filename):
        self.filename = filename
        self.size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self.size) * 100
            logger.info("\r%s %s / %s  (%.2f%%)" % (
                self.filename, self._seen_so_far, self.size,
                percentage), end="")


class CloudStorage:
    def __init__(self, access_key=None, secret_key=None, endpoint_url=None, bucket=None, session_token=None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.client = None
        self.session_token = session_token

    def initialize(self):
        if self.access_key is None or self.secret_key is None or self.endpoint_url is None:
            logger.error("Please provide access_key, secret_key, session_token and endpoint_url")
            raise
        self.client = boto3.client('s3',
                                   endpoint_url=self.endpoint_url,
                                   aws_access_key_id=self.access_key,
                                   aws_secret_access_key=self.secret_key,
                                   aws_session_token=self.session_token
                                   )
        return self

    def upload_folder(self, local_folder, cloud_path, bucket=None):
        if bucket is None and self.bucket is None:
            logger.error("Please provide bucket name")
            return
        if bucket is None:
            bucket = self.bucket
        stream = tqdm(os.walk(local_folder))
        for root, dirs, files in stream:
            for file in files:
                localFilePath = os.path.join(root, file)
                relativePath = os.path.relpath(localFilePath, local_folder)
                cloudPath = os.path.join(cloud_path, relativePath)
                cloudPath = cloudPath.replace('\\', '/')
                try:
                    self.client.upload_file(localFilePath, bucket, cloudPath,
                                            Callback=ProgressPercentage(localFilePath))
                except botocore.exceptions.ClientError as e:
                    logger.error(e)
