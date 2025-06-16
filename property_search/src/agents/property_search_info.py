import boto3
import json
import os
from boto3.dynamodb.conditions import Key

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb')
property_table_name = os.getenv('PROPERTY_TABLE', 'Property_Catalog')
table = dynamodb.Table(property_table_name)

def get_named_parameter(event, name):
    """Extract a named parameter from the event"""
    return next(item for item in event['parameters'] if item['name'] == name)['value']

def populate_function_response(event, response_body):
    """Format the response for Bedrock Agent"""
    return {
        'response': {
            'actionGroup': event['actionGroup'],
            'function': event['function'],
            'functionResponse': {
                'responseBody': {
                    'TEXT': {
                        'body': json.dumps(response_body)
                    }
                }
            }
        }
    }

def get_property_id_by_address(address):
    """Get property ID using address"""
    try:
        response = table.query(
            IndexName='AddressIndex',
            KeyConditionExpression=Key('address').eq(address)
        )
        return response['Items'][0]['property_id'] if response['Items'] else None
    except Exception as e:
        return {'error': str(e)}

def get_property_by_id(property_id):
    """Retrieve full property data by ID"""
    try:
        response = table.get_item(Key={'property_id': property_id})
        return json.loads(response['Item']['property_data']) if 'Item' in response else {}
    except Exception as e:
        return {'error': str(e)}

def extract_property_core(data):
    """Extract core property details"""
    return {
        'price': data['property']['price'],
        'bedrooms': data['property']['bedrooms'],
        'bathrooms': data['property']['bathrooms'],
        'square_footage': data['property']['square_footage'],
        'year_built': data['property']['year_built'],
        'parking': data['property']['parking'],
        'garage_spaces': data['property']['garage_spaces'],
        'stories': data['property']['stories']
    }

def extract_climate(data):
    """Extract climate risk information"""
    return data['climate']

def extract_neighborhood(data):
    """Extract neighborhood details"""
    return data['neighborhood']

def extract_sales_tax_history(data):
    """Extract sales and tax history"""
    return data['sale_and_tax_history']

def extract_exterior(data):
    """Extract exterior features"""
    return {
        'construction': data['property']['exterior']['construction'],
        'yard_features': data['property']['yard']['features'],
        'patio': data['property']['exterior']['patio']
    }

def extract_interior(data):
    """Extract interior features"""
    return data['property']['interior_features']

def extract_kitchen(data):
    """Extract kitchen details"""
    return data['property']['kitchen']

def extract_hoa_utilities(data):
    """Extract HOA and utilities information"""
    return {
        'hoa': data['property']['hoa'],
        'utilities': data['property']['utilities']
    }

def lambda_handler(event, context):
    """Main Lambda handler"""
    print("Received event:", json.dumps(event, indent=2))
    
    function = event.get('function', '')
    parameters = event.get('parameters', [])
    result = {}

    try:
        if function == 'get_property_id_by_address':
            address = get_named_parameter(event, 'address')
            result = {'property_id': get_property_id_by_address(address)}
            
        elif function == 'get_property_core':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_property_core(data)
            
        elif function == 'get_climate_info':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_climate(data)
            
        elif function == 'get_neighborhood_info':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_neighborhood(data)
            
        elif function == 'get_sales_tax_history':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_sales_tax_history(data)
            
        elif function == 'get_exterior_details':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_exterior(data)
            
        elif function == 'get_interior_details':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_interior(data)
            
        elif function == 'get_kitchen_details':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_kitchen(data)
            
        elif function == 'get_hoa_utilities':
            property_id = get_named_parameter(event, 'property_id')
            data = get_property_by_id(property_id)
            result = extract_hoa_utilities(data)
            
        else:
            result = {'error': f'Unknown function: {function}'}

    except Exception as e:
        result = {'error': str(e)}
        print(f"Error processing request: {str(e)}")

    return populate_function_response(event, result)
