import boto3
import os

local_path = 'tiny_bert_sentiment_analysis'

bucket_name = 'prjsentimentanalysis'

s3 = boto3.client('s3')

def download_dir(local_path,mod_name):
    s3_prefix = 'ml_model/'+mod_name
    os.makedirs(local_path,exist_ok= True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket = bucket_name,Prefix =s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                relative_path = s3_key[len(s3_prefix):]  # Remove the prefix from the S3 key to get relative path
                local_file = os.path.join(local_path, relative_path)
                local_dir = os.path.dirname(local_file)
                
                os.makedirs(local_dir, exist_ok=True)  # Create directories as needed
                
                # Download the file from S3
                s3.download_file(bucket_name, s3_key, local_file)