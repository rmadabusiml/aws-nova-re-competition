import os
import json
import csv
import boto3
from glob import glob
from botocore.exceptions import ClientError
from datetime import datetime
from helper.utils import flatten_paragraphs, upload_file_to_s3

class PropertyDescriptionGenerator:
    def __init__(self, s3_bucket, bedrock_model='amazon.nova-lite-v1:0'):
        self.s3 = boto3.client('s3')
        self.bedrock = boto3.client('bedrock-runtime')
        self.bucket = s3_bucket
        self.model_id = bedrock_model

        self.text_dir = '../../data/property_descriptions'
        os.makedirs(self.text_dir, exist_ok=True)
        
    def _generate_description(self, property_data):
        """Invoke Nova model to generate property description"""
        system_prompt = """Act as a real estate description specialist. Transform the provided JSON property data into a natural language description optimized for user searches. Follow these guidelines:

            1. **Structure**:
            - Opening Hook: Start with "This [property type] at [address] offers..." 
            - Key Stats: Square footage, bed/bath count, year built
            - Interior Highlights: Focus on kitchen features, flooring, smart tech
            - Exterior/Community: Yard details, amenities, HOA info
            - Location Perks: Schools (name/distance/rating), transportation access
            - Unique Selling Points: Recent upgrades, energy efficiency

            2. **Technical Requirements**:
            - Include exact distances (0.6 miles to X School)
            - List specific appliances/models when available
            - Note construction materials (HardiPlank siding)
            - Mention energy certifications (ENERGY STAR)
            - Add climate risk scores if present

            3. **Search Optimization**:
            - Embed natural synonyms ("chef's kitchen" + "gourmet cooking space")
            - Use neighborhood colloquialisms ("10 mins from downtown" vs "2.3 miles")
            - Include 3-5 hidden search terms from: [smart home], [low maintenance], [commuter friendly], [turnkey], [investment property]

            4. **Format**:
            - 12-15 sentences
            - 400-600 words
            - Paragraph breaks every 2-3 features
            - Bold key terms sparingly
            - Avoid markdown

            5. **Special Instructions**:
            - If square footage >2500, mention "spacious for entertaining"
            - If built post-2020, highlight "modern building standards"
            - For HOA fees >$500/mo, note "premium amenities package"
            - Flag flood risk >10% prominently

            Example Output:
            "This modern 4-bedroom home at 632 Otto Ave offers 2,067 sqft of single-story living built in 2021. The chef's kitchen features granite counters, gas range, and oversized pantry, flowing into a smart home-enabled living area with 9' ceilings. Enjoy low-maintenance landscaping with irrigation and a 3-car garage with epoxy floors. Part of Fairhaven community with pool and trails, just 0.6 miles from Williams Elementary (rated 8/10). Recent upgrades include 2025 roof replacement and whole-house water softener. Ideal for commuters - 8 mins to I-35 with EV charging ready."

            Process the following JSON:
            """  

        messages = [{
            "role": "user",
            "content": [{
                "text": f"Process the following JSON:\n{json.dumps(property_data)}"
            }]
        }]

        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": system_prompt}],
                inferenceConfig={
                    "maxTokens": 4096,
                    "temperature": 0.3,
                    "topP": 0.8
                }
            )
            
            return response['output']['message']['content'][0]['text']
        except ClientError as e:
            print(f"Bedrock API error: {e.response['Error']['Message']}")
            return "Description generation failed"

    def _store_processed_data(self, property_id, description, data, source_file):
        """Store generated data and upload original JSON"""
        
        text_filename = f"{property_id}.txt"
        text_metadata_filename = f"{property_id}.txt.metadata.json"
        local_text_path = os.path.join(self.text_dir, text_filename)
        local_text_metadata_path = os.path.join(self.text_dir, text_metadata_filename)
        
        with open(local_text_path, 'w') as text_file:
            text_file.write(description)

        metadata ={}
        address_parts = data['property']['address'].split(', ')

        metadata["property_id"] = property_id
        metadata["city"] = address_parts[-2].strip()
        metadata["state"] = address_parts[-1].split()[0]
        metadata["address"] = data['property']['address']
        metadata["price"] = data['property']['price']
        metadata["square_footage"] = data['property']['square_footage']
        metadata["year_built"] = data['property']['year_built']

        json_data = {"metadataAttributes": metadata}

        with open(local_text_metadata_path, "w") as f:
            json.dump(json_data, f)
        
        try:
            upload_file_to_s3(self.s3, self.bucket, source_file, f'real_estate/property_search/raw/{os.path.basename(source_file)}')
            upload_file_to_s3(self.s3, self.bucket, local_text_path, f'real_estate/property_search/processed/{text_filename}')
            upload_file_to_s3(self.s3, self.bucket, local_text_metadata_path, f'real_estate/property_search/processed/{text_metadata_filename}')
        except ClientError as e:
            print(f"S3 upload error: {e.response['Error']['Message']}")

    def process_directory(self, input_dir):
        """Process all JSON files in a directory"""
        # print("Processing directory: ", input_dir)
        for json_file in glob(os.path.join(input_dir, '*.json')):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    property_id = os.path.splitext(os.path.basename(json_file))[0]
                    print(property_id)
                    
                    description = self._generate_description(data)
                    description = flatten_paragraphs(description)
                    print(description)
                    
                    self._store_processed_data(property_id, description, data, json_file)
                    
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue

if __name__ == '__main__':
    processor = PropertyDescriptionGenerator(s3_bucket='handsonllms-raghu')
    processor.process_directory('../../data/property_details/GT/json')
    processor.process_directory('../../data/property_details/AT/json')
    processor.process_directory('../../data/property_details/IN/json')
