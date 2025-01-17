from google.cloud import storage
import os
import json

class GCPStorageManager:
    def __init__(self, credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.storage_client = storage.Client()

    def create_bucket(self, bucket_name, storage_class='STANDARD', location='us-central1'):
        """Create a new bucket in specific location with storage class"""
        bucket = self.storage_client.bucket(bucket_name)
        bucket.storage_class = storage_class
        new_bucket = self.storage_client.create_bucket(bucket, location=location)
        return f"Bucket {new_bucket.name} created in {new_bucket.location} with storage class {new_bucket.storage_class}"

    def upload_file(self, bucket_name, source_file_name, destination_file_name):
        """Uploads a file to the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_file_name)
        blob.upload_from_filename(source_file_name)
        return f"File {source_file_name} uploaded to {destination_file_name}."

    def upload_content(self, bucket_name, content, destination_file_name, content_type="application/pdf"):
        """Uploads content directly to the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_file_name)
        blob.upload_from_string(content, content_type=content_type)
        # Make the blob public
        blob.make_public()
        return blob.public_url

    def list_files(self, bucket_name):
        """List all files in the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        files = bucket.list_blobs()
        file_names = [file.name for file in files]
        return file_names

    def download_file(self, bucket_name, source_file_name, destination_file_name):
        """Downloads a file from the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_file_name)
        blob.download_to_filename(destination_file_name)
        return f"File {source_file_name} downloaded to {destination_file_name}."

    def download_content(self, bucket_name, source_file_name):
        """Downloads content from the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_file_name)
        content = blob.download_as_bytes()
        return content

    def upload_json(self, bucket_name, data, destination_file_name):
        """Uploads a JSON object to the bucket."""
        json_content = json.dumps(data)
        return self.upload_content(bucket_name, json_content, destination_file_name, content_type="application/json")

    def download_json(self, bucket_name, source_file_name):
        """Downloads a JSON object from the bucket."""
        content = self.download_content(bucket_name, source_file_name)
        return json.loads(content.decode('utf-8'))
    
    def file_exists(self, bucket_name, file_name):
        """Checks if a file exists in the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        return blob.exists()

    def create_folder(self, bucket_name, folder_name):
        """Creates a folder in the bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(folder_name + '/')
        blob.upload_from_string('')
        return f"Folder {folder_name} created in {bucket_name}."