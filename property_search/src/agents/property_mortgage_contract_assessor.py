import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
s3_client = boto3.client("s3")

MODEL_ID = os.environ.get("mortgage_contract_llm", "amazon.nova-pro-v1:0")
BUCKET = "handsonllms-raghu"

EXTRACT_MODEL_ID = os.environ.get('property_appraiser_llm', 'amazon.nova-lite-v1:0')

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

def get_pdf_bytes(key):
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    return response["Body"].read()

def lambda_handler(event, context):
    try:

        property_id = extract_property_id(event.get('query', ''), event.get('chatHistory', []))
        logger.info(f"Extracted property id: {property_id}")

        if not property_id:
            return {
                "body": json.dumps({"response": "Could not identify valid property id in query"})
            }

        contract_f_key = f'real_estate/mortgage_contracts/{property_id}_F.pdf'
        contract_r_key = f'real_estate/mortgage_contracts/{property_id}_R.pdf'

        doc_f = get_pdf_bytes(contract_f_key)
        doc_r = get_pdf_bytes(contract_r_key)

        prompt_text = """
        Compare the two mortgage contracts and provide the following:
        1. Which contract is more borrower-friendly and why.
        2. Highlight key differences in the following clauses:
        - Prepayment penalties
        - Escrow management
        - Default handling and grace periods
        - Dispute resolution process
        - Recourse and personal liability terms
        3. Summarize the pros and cons of each contract such as which contract is more borrower-friendly and why. Ensure to include the lender name in the response.
        """

        messages = [{
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": "pdf",
                        "name": "FriendlyContract",
                        "source": {"bytes": doc_f}
                    }
                },
                {
                    "document": {
                        "format": "pdf",
                        "name": "RestrictiveContract",
                        "source": {"bytes": doc_r}
                    }
                },
                {
                    "text": prompt_text
                }
            ]
        }]

        response = bedrock_runtime.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 8192,
                "temperature": 0.4,
                "topP": 0.9
            }
        )

        output_text = response["output"]["message"]["content"][0]["text"]
        logger.info(f"Comparison Result: {output_text}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': output_text,
            })
        }

    except Exception as e:
        logger.error(f"Lambda error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
