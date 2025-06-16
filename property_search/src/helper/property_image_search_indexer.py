import boto3
import base64
import json
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os

BUCKET_NAME = os.environ.get('IMAGE_BUCKET', 'handsonllms-raghu')
SUMMARY_MODEL_ID = os.environ.get('property_image_description_llm', 'amazon.nova-lite-v1:0')
EMBEDDING_MODEL_ID = os.environ.get('property_image_embedding_llm', 'amazon.titan-embed-image-v1')

host = '1hu0dtumr17dixj14ht6.us-east-1.aoss.amazonaws.com'
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, 'us-east-1', 'aoss')

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

os_client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )

def generate_embedding(image_bytes):
    response = bedrock.invoke_model(
        body=json.dumps({
            "inputImage": base64.b64encode(image_bytes).decode('utf8'),
            "embeddingConfig": {"outputEmbeddingLength": 1024}
        }),
        modelId=EMBEDDING_MODEL_ID
    )
    return json.loads(response['body'].read())['embedding']

def generate_description(image_bytes):
    """Generate text description using Amazon Nova"""
    print(f"Generating description for image")
    system_prompt = [{"text": "You are a real estate property description expert. Describe the property exterior features in 2 sentences about architectural style, materials, and distinctive features."}]
    
    response = bedrock.converse(
        modelId=SUMMARY_MODEL_ID,
        system=system_prompt,
        messages=[{
            'role': 'user',
            'content': [{
                'image': {'format': 'png', 'source': {'bytes': image_bytes}}
            }]
        }]
    )
    # return response['output']['message']['content']
    return response['output']['message']['content'][0]['text']

def index_images():
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix='real_estate/property_images'):
        for obj in page.get('Contents', []):
            if not obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            try:
                img_bytes = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])['Body'].read()
                embedding = generate_embedding(img_bytes)
                description = generate_description(img_bytes)

                print(f"Generating embedding and description for {obj['Key']}")
                print(f"Description: {description}")
                
                document = {
                    "s3_key": obj['Key'],
                    "embedding": embedding,
                    "description": description
                }
                
                # Auto-generate document ID
                os_client.index(
                    index="property-search-images",
                    body=document
                )

                print(f"Indexed {obj['Key']}")
                
            except Exception as e:
                print(f"Error indexing {obj['Key']}: {str(e)}")

if __name__ == '__main__':
    index_images()
