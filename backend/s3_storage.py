import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        
        # Initialize S3 Client
        # It automatically picks up credentials from env vars or IAM role
        self.s3 = boto3.client(
            's3',
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    def upload_file(self, file_obj, object_name):
        """Upload a file-like object to S3"""
        try:
            self.s3.upload_fileobj(file_obj, self.bucket_name, object_name)
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
            logger.info(f"File uploaded to S3: {url}")
            return url
        except Exception as e:
            logger.error(f"S3 Upload Error: {str(e)}")
            raise

    def download_file(self, object_name, local_path):
        """Download file from S3 to local path (temporary)"""
        try:
            self.s3.download_file(self.bucket_name, object_name, local_path)
            logger.info(f"File downloaded from S3 to {local_path}")
        except Exception as e:
            logger.error(f"S3 Download Error: {str(e)}")
            raise

    def delete_file(self, object_name):
        """Delete file from S3"""
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(f"File deleted from S3: {object_name}")
        except Exception as e:
            logger.error(f"S3 Delete Error: {str(e)}")
            raise
