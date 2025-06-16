import base64
import json
import boto3
import logging
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET_NAME = os.environ.get('IMAGE_BUCKET', 'handsonllms-raghu')
SUMMARY_MODEL_ID = os.environ.get('property_image_description_llm', 'amazon.nova-lite-v1:0')
EXTRACT_MODEL_ID = os.environ.get('property_image_extract_llm', 'amazon.nova-lite-v1:0')
EMBEDDING_MODEL_ID = os.environ.get('property_image_embedding_llm', 'amazon.titan-embed-image-v1')
OPENSEARCH_HOST = os.environ.get('OPENSEARCH_HOST', '1hu0dtumr17dixj14ht6.us-east-1.aoss.amazonaws.com')
REGION = 'us-east-1'

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name=REGION)
session = boto3.Session()
credentials = session.get_credentials()
auth = AWSV4SignerAuth(credentials, REGION, 'aoss')

os_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

def extract_image_path(query, chat_history):
    """Use Nova Micro model to extract image path from query"""
    try:
        response = bedrock.invoke_model(
            modelId=EXTRACT_MODEL_ID,
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "text": f"""Extract the image path from this query and return only the image path (eg: real_estate/property_images/input/1234.png) without additional details. Query: {query}. Chat History: {chat_history}"""
                    }]
                }]
            })
        )
        return json.loads(response['body'].read())['output']['message']['content'][0]['text'].strip()
        
    except Exception as e:
        logger.error(f"ID extraction error: {str(e)}")
        return None

def generate_embedding(image_bytes):
    """Generate image embedding using Amazon Titan Multimodal"""
    response = bedrock.invoke_model(
        body=json.dumps({
            "inputImage": base64.b64encode(image_bytes).decode('utf8'),
            "embeddingConfig": {"outputEmbeddingLength": 1024}
        }),
        modelId=EMBEDDING_MODEL_ID
    )
    return json.loads(response['body'].read())['embedding']

def search_opensearch(query_embedding):
    """Search OpenSearch Serverless using k-NN"""
    search_body = {
        "size": 2,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 6
                }
            }
        }
    }
    
    response = os_client.search(
        index="property-search-images",
        body=search_body
    )
    return response['hits']['hits']

def lambda_handler(event, context):
    try:
        query_image_key = extract_image_path(event.get('query', ''), event.get('chat_history', ''))
        logger.info(f"Query image key: {query_image_key}")

        query_bytes = s3.get_object(Bucket=BUCKET_NAME, Key=query_image_key)['Body'].read()
        query_embedding = generate_embedding(query_bytes)
        results = search_opensearch(query_embedding)
        
        output = []
        for hit in results:
            s3_key = hit['_source']['s3_key']
            logger.info(s3_key)

            
            output.append({
                "image_url": s3_key
            })

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': json.dumps(output)
            })
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }