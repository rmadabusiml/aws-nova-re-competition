import base64
import json
import boto3
from botocore.exceptions import ClientError
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
EXTRACT_MODEL_ID = os.environ.get('property_appraiser_llm', 'amazon.nova-lite-v1:0')
SUMMARY_MODEL_ID = os.environ.get('property_appraiser_llm', 'amazon.nova-pro-v1:0')
s3_client = boto3.client('s3')

def extract_property_id(query, chat_history):
    """Use Nova Micro model to extract property id from query"""
    try:
        response = bedrock_runtime.invoke_model(
            modelId=EXTRACT_MODEL_ID,
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "text": f"""Extract the property id from this query and return only the property id (eg: 12345678) without additional details. Query: {query}. Chat History: {chat_history}"""
                    }]
                }]
            })
        )
        return json.loads(response['body'].read())['output']['message']['content'][0]['text'].strip()
        
    except Exception as e:
        logger.error(f"ID extraction error: {str(e)}")
        return None

def lambda_handler(event, context):
    try:
        
        logger.info(f"event: {event}")
        query = event.get('query', '')
        chat_history = event.get('chatHistory', [])
        logger.info(f"chat_history: {chat_history}")

        property_id = extract_property_id(query, chat_history)
        logger.info(f"Extracted property id: {property_id}")

        if not property_id:
            return {
                "body": json.dumps({"response": "Could not identify valid property id in query"})
            }

    
        response = s3_client.get_object(
            Bucket='handsonllms-raghu',
            Key=f'real_estate/inspection_reports/inspection_report_summary_{property_id}.pdf'
        )

        doc_bytes = response['Body'].read()

        messages = [{
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": "pdf",
                        "name": "InspectionReport",
                        "source": {
                            "bytes": doc_bytes
                        }
                    }
                },
                {
                    "text": """Analyze this home inspection report and provide:
                    1. Fannie Mae Property Condition Rating (C1-C5) with explanation
                    2. Fannie Mae Property Quality Rating (Q1-Q5) with explanation
                    3. Repair cost breakdown for each major issue
                    4. Total estimated repair cost"""
                }
            ]
        }]

        inf_params = {
            "maxTokens": 8192,
            "topP": 0.9,
            "temperature": 0.5
        }

        model_response = bedrock_runtime.converse(
            modelId=SUMMARY_MODEL_ID,
            messages=messages,
            inferenceConfig=inf_params
        )

        output_text = model_response['output']['message']['content'][0]['text']
        logger.info(f"Model response: {output_text}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': output_text,
                'property_id': property_id
            })
        }

    except ClientError as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f"AWS Error: {str(e)}")
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
