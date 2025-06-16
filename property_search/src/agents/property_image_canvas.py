import base64
import json
import boto3
import os
import logging
from random import randint
from io import BytesIO
import uuid
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)


s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

BUCKET_NAME = os.environ.get('IMAGE_BUCKET', 'handsonllms-raghu')
EXTRACT_MODEL_ID = os.environ.get('property_image_extract_llm', 'amazon.nova-lite-v1:0')
IMAGE_GEN_MODEL_ID = os.environ.get('NOVA_CANVAS_MODEL_ID', 'amazon.nova-canvas-v1:0')

def download_image(s3_key):
    """Download image from S3 and return bytes"""
    response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    return response['Body'].read()

def upload_image(image_bytes, s3_key):
    """Upload image bytes to S3"""
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=image_bytes,
        ContentType='image/png'
    )
    return s3_key

def extract_json_block(text):
    # Use regex to find the block between ```json and ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)  # parse as dict
    else:
        return None

def extract_image_task_info(query, chat_history):
    """Extract s3_key, task_type, user_prompt, and mask_prompt using Nova"""
    try:
        extraction_prompt = f"""Analyze this real estate image processing request and extract:
        1. S3 key path (format: real_estate/property_images/input/...)
        2. Task type: TEXT_IMAGE (change color or texture) or INPAINTING (replace objects or modernize some objects in the image)
        3. Main user prompt (the desired change)
        4. Mask prompt if INPAINTING (area to modify)

        Return JSON format. Examples:
        Example 1 Input: "Based on real_estate/property_images_generation/input/kitchen_1.jpg, update the kitchen to have modern stainless steel appliances"
        Output: {{"s3_key": "real_estate/property_images/input/kitchen_1.jpg", "task_type": "TEXT_IMAGE", "user_prompt": "update the kitchen to have modern stainless steel appliances", "mask_prompt": null}}

        Example 2 Input: "Based on real_estate/property_images_generation/input/backyard_3.png, replace the wooden fence with a stone wall on the right side"
        Output: {{"s3_key": "real_estate/property_images/input/backyard_3.png", "task_type": "INPAINTING", "user_prompt": "replace the wooden fence with a stone wall on the right side", "mask_prompt": "wooden fence on the right side"}}

        Current Input: {query}
        Chat History: {chat_history}
        """

        response = bedrock.invoke_model(
            modelId=EXTRACT_MODEL_ID,
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [{
                    "role": "user",
                    "content": [{"text": extraction_prompt}]
                }]
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        content = response_body['output']['message']['content']
        
        if isinstance(content, list) and len(content) > 0:
            first_content = content[0]
            if 'text' in first_content:
                raw_output = first_content['text'].strip()
            else:
                raise ValueError("No text content in response")
        else:
            raise ValueError("Unexpected response format from Bedrock")

        cleaned_output = extract_json_block(raw_output)
        result = cleaned_output
        
        return {
            "s3_key": result.get("s3_key"),
            "task_type": result.get("task_type", "TEXT_IMAGE"),
            "user_prompt": result.get("user_prompt", ""),
            "mask_prompt": result.get("mask_prompt")
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        return {"error": "Failed to parse model response"}
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return {"error": str(e)}


def generate_conditioned_image(source_image_b64, prompt):
    """Generate image guided by source image and text prompt"""
    logger.info(f"Generating conditioned image for {prompt}")
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "conditionImage": source_image_b64,
            "controlMode": "SEGMENTATION",
            "controlStrength": 0.8
        },
        "imageGenerationConfig": {
            "numberOfImages": 2,
            "quality": "standard",
            "width": 1280,
            "height": 720,
            "cfgScale": 8.0,
            "seed": randint(0, 858993459)
        }
    }
    
    response = bedrock.invoke_model(
        modelId=IMAGE_GEN_MODEL_ID,
        body=json.dumps(body)
    )
    
    return json.loads(response['body'].read())

def generate_inpainting_mask_prompt(source_image_b64, mask_prompt, replacement_prompt):
    """Replace objects using text-based mask prompt"""
    logger.info(f"Generating inpainting image for {mask_prompt}")
    body = {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "image": source_image_b64,
            "maskPrompt": mask_prompt,
            "text": replacement_prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 2,
            "height": 720,
            "width": 1280,
            "quality": "standard",
            "cfgScale": 8.0,
            "seed": randint(0, 858993459)
        }
    }
    
    response = bedrock.invoke_model(
        modelId=IMAGE_GEN_MODEL_ID,
        body=json.dumps(body)
    )
    return json.loads(response['body'].read())

def process_generation(event, extraction_result):
    """Handle different generation types based on event parameters"""
    logger.info(f"Processing generation for {extraction_result}")
    source_key = extraction_result['s3_key']
    source_bytes = download_image(source_key)
    source_b64 = base64.b64encode(source_bytes).decode('utf-8')
    
    generated_keys = []

    if extraction_result['task_type'] == 'TEXT_IMAGE':
        response = generate_conditioned_image(source_b64, extraction_result['user_prompt'])
        for idx, output in enumerate(response.get('images', [])):
            img_bytes = base64.b64decode(output)
            new_key = f"real_estate/property_images/output/{uuid.uuid4()}.png"
            upload_image(img_bytes, new_key)
            generated_keys.append(new_key)

    if extraction_result['task_type'] == 'INPAINTING':
        response = generate_inpainting_mask_prompt(source_b64, extraction_result['mask_prompt'], extraction_result['user_prompt'])
        for idx, output in enumerate(response.get('images', [])):
            img_bytes = base64.b64decode(output)
            new_key = f"real_estate/property_images/output/{uuid.uuid4()}.png"
            upload_image(img_bytes, new_key)
            generated_keys.append(new_key)
    
    return generated_keys

def lambda_handler(event, context):
    try:
        extraction_result = extract_image_task_info(
            event.get('query', ''),
            event.get('chat_history', '')
        )
        
        if 'error' in extraction_result:
            return {'statusCode': 400, 'body': json.dumps(extraction_result)}

        generated_keys = process_generation(event, extraction_result)
        logger.info(f"Generated keys: {generated_keys}")
            
        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': generated_keys,
                'message': 'Successfully processed image generation'
            })
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }