import boto3
import json
import os
from glob import glob
from botocore.exceptions import ClientError

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

def create_property_catalog_table():
    try:
        table = dynamodb.create_table(
            TableName='Property_Catalog',
            KeySchema=[{'AttributeName': 'property_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[
                {'AttributeName': 'property_id', 'AttributeType': 'S'},
                {'AttributeName': 'address', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'AddressIndex',
                    'KeySchema': [{'AttributeName': 'address', 'KeyType': 'HASH'}],
                    'Projection': {'ProjectionType': 'KEYS_ONLY'},
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.wait_until_exists()
        return table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            return dynamodb.Table('Property_Catalog')
        else:
            raise

def insert_property_data(table, file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    property_id = file_path.split('/')[-1].split('.')[0]
    address = data['property']['address']
    
    item = {
        'property_id': property_id,
        'address': address,
        'property_data': json.dumps(data)
    }
    
    table.put_item(Item=item)
    return property_id

def process_property_directory(input_dir):
    table = dynamodb.Table('Property_Catalog')
    
    for json_file in glob(os.path.join(input_dir, '*.json')):
        try:
            insert_property_data(table, json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue

def extract_property_core(data):
    prop = data['property']
    return {
        'price': prop['price'],
        'bedrooms': prop['bedrooms'],
        'bathrooms': prop['bathrooms'],
        'square_footage': prop['square_footage'],
        'year_built': prop['year_built'],
        'stories': prop['stories'],
        'parking': prop['parking'],
        'garage_spaces': prop['garage_spaces'],
        'floors': prop['floors'],
        'floor_details': prop['floor_details']
    }

def extract_climate(data):
    return data['climate']

def extract_neighborhood(data):
    return data['neighborhood']

def extract_sales_tax_history(data):
    return data['sale_and_tax_history']

def extract_interior_features(data):
    return data['property']['interior_features']

def extract_exterior(data):
    return {
        'construction': data['property']['exterior']['construction'],
        'yard_features': data['property']['yard']['features'],
        'patio': data['property']['exterior']['patio']
    }

def extract_kitchen(data):
    return data['property']['kitchen']

def extract_hoa_utilities(data):
    return {
        'hoa': data['property']['hoa'],
        'utilities': data['property']['utilities']
    }

# 4. Query Functions
def get_property_by_id(table, property_id):
    response = table.get_item(Key={'property_id': property_id})
    if 'Item' in response:
        return json.loads(response['Item']['property_data'])
    return None

def get_property_id_by_address(table, address):
    response = table.query(
        IndexName='AddressIndex',
        KeyConditionExpression='address = :addr',
        ExpressionAttributeValues={':addr': address}
    )
    return response['Items'][0]['property_id'] if response['Items'] else None

if __name__ == '__main__':
    # create_property_catalog_table()
    # insert_property_data('../../data/property_details/GT/json_sample/32889208.json')

    process_property_directory('../../data/property_details/GT/json')
    process_property_directory('../../data/property_details/AT/json')
    process_property_directory('../../data/property_details/IN/json')

    # Query example
    # address = "30418 La Quinta Dr, Georg, TX 78628"
    # table = dynamodb.Table('Property_Catalog')
    
    # # Get property ID from address
    # pid = get_property_id_by_address(table, address)
    # if pid:
    #     # Get full data
    #     property_data = get_property_by_id(table, pid)
        
    #     # Extract specific components
    #     core_details = extract_property_core(property_data)
    #     kitchen_details = extract_kitchen(property_data)
    #     climate_info = extract_climate(property_data)
        
    #     print("Core Details:", core_details)
    #     print("\nKitchen Features:", kitchen_details['features'])
    #     print("\nClimate Risks:", climate_info)
    