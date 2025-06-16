import os
import sys
import time
import boto3
import logging
import pprint
import json

from knowledge_base import BedrockKnowledgeBase

# ------------------ Logging Setup ------------------ #
logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------ AWS Clients ------------------ #
region = "us-east-1"
s3_client = boto3.client('s3')
bedrock_agent_client = boto3.client('bedrock-agent', region_name=region)
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name=region)

# ------------------ Constants ------------------ #
KNOWLEDGE_BASE_NAME = 're-prop-desc-kb'
KNOWLEDGE_BASE_DESCRIPTION = "Knowledge Base for real estate property description."
BUCKET_NAME = 'handsonllms-raghu'
DATA_PREFIX = 'real_estate/property_search/processed/'

# ------------------ Functions ------------------ #
def generate_suffix_from_timestamp():
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))[-7:]
    return timestamp_str

def create_knowledge_base(suffix):
    kb_name = f'{KNOWLEDGE_BASE_NAME}-{suffix}'
    data_source = [{
        "type": "S3",
        "bucket_name": BUCKET_NAME,
        "prefix": DATA_PREFIX
    }]

    kb_metadata = BedrockKnowledgeBase(
        kb_name=kb_name,
        kb_description=KNOWLEDGE_BASE_DESCRIPTION,
        data_sources=data_source,
        chunking_strategy="NONE",
        suffix=suffix
    )

    return kb_metadata

def sync_knowledge_base(kb_metadata):
    logger.info("Waiting 30 seconds to ensure Knowledge Base is available...")
    time.sleep(30)
    kb_metadata.start_ingestion_job()
    kb_id = kb_metadata.get_knowledge_base_id()
    return kb_id

def main():
    logger.info("Starting Knowledge Base setup...")
    
    suffix = generate_suffix_from_timestamp()
    kb_metadata = create_knowledge_base(suffix)
    kb_id = sync_knowledge_base(kb_metadata)

    logger.info(f"Knowledge Base ID: {kb_id}")
    print("Knowledge Base ID:", kb_id)

# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    main()
