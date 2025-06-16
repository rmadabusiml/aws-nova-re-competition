import boto3

def flatten_paragraphs(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ' '.join(lines)

def upload_file_to_s3(s3_client, bucket, file_path, s3_key):
    try:
        s3_client.upload_file(Filename=file_path, Bucket=bucket, Key=s3_key)
        print(f"Uploaded '{file_path}' to 's3://{bucket}/{s3_key}'")
    except Exception as e:
        print(f"Error uploading file: {e}")