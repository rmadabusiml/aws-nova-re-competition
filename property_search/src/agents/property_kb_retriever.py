import boto3
import json
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
foundation_model = os.environ.get("property_agent_llm", "amazon.nova-pro-v1:0")
knowledge_base_id = os.environ.get("property_agent_knowledge_base_id", "IBTJXPOYRN")

def retrieve_and_generate(query):
    response = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": knowledge_base_id,
                    "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{foundation_model}",
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 10,
                        }
                    },
                    "generationConfiguration": {
                        "guardrailConfiguration": {
                            "guardrailId": "3pt5fpo4foxt",
                            "guardrailVersion": "1"
                        }
                    }
                }
            }
        )

    return response

def get_metadata_from_citations(response):
    properties = []
    for citation in response.get('citations', []):
        for ref in citation.get('retrievedReferences', []):
            metadata = ref.get('metadata', {})
            prop = {
                "address": metadata.get('address', 'N/A'),
                "property_id": metadata.get('property_id', 'N/A')
            }
            if prop["address"] != 'N/A' and prop["property_id"] != 'N/A':
                properties.append(prop)

    return properties

def lambda_handler(event, context):
    try:
        # Parse input
        logger.info(f"event: {event}")
        query = event.get('query', '')
        chat_history = event.get('chatHistory', [])
        logger.info(f"chat_history: {chat_history}")

        response = retrieve_and_generate(query)
        properties = get_metadata_from_citations(response)

        return {
            "body": json.dumps({
                "response": properties
            })
        }
    except Exception as e:
        logger.error(f"Lambda execution error: {str(e)}")
        return {
            "body": json.dumps({"response": "Error processing your request"})
        }