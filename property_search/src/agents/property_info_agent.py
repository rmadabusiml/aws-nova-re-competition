import os
import boto3
import json
import uuid
from helper.bedrock_agent_helper import AgentsForAmazonBedrock

# Initialize the Bedrock Agent helper
agents = AgentsForAmazonBedrock()

# Generate unique ID for resources
resource_suffix = str(uuid.uuid4())[:8]

# Agent configuration
property_agent_name = f"property-agent-{resource_suffix}"
property_lambda_name = f"fn-property-agent-{resource_suffix}"

# DynamoDB table configuration
property_table = 'Property_Catalog'
dynamoDB_args = None
# dynamoDB_args = [property_table, 'property_id']

# Get AWS account ID and region
account_id = boto3.client("sts").get_caller_identity()["Account"]
region = agents.get_region()

# Agent foundation model configuration
agent_foundation_model = [os.environ.get('property_agent_llm', 'amazon.nova-pro-v1:0')]

# Agent description and instructions
agent_description = "You are a Property Assistant that helps users find and analyze real estate properties."
agent_instruction = """

You are a Property Assistant that helps users find and analyze real estate properties.
Your capabilities include:
1. Retrieving property ID by physical address. The provided address is typically in the format of street_number street_name, city, state (2 letter) zipcode such as 1333 Street Name, City Name, TX 78633. The address may not have commas always but treat them as address as long as they have specific parts of an address.
2. Providing core details (price, size, bedrooms/bathrooms)
3. Analyzing climate risk factors (flood, fire, heat)
4. Showing neighborhood characteristics and school information
5. Displaying sales history and tax records
6. Detailing exterior features (construction, yard, patio)
7. Listing interior amenities and layout details 
8. Providing kitchen specifications and appliances
9. Explaining HOA rules and utility configurations

Core behaviors:
1. First verify property ID through address lookup when needed. Use the full address provided by the user and not partial.
2. Access all available data systems before requesting additional information
3. Maintain professional tone while avoiding technical real estate jargon
4. Present complex data in digestible visual-friendly formats

Response style:
- Prioritize accuracy over speed for financial/legal information
- Use bullet points for multi-item responses
- Highlight key decision factors in bold
- Keep explanations under 5 sentences
- Separate factual data from analytical insights

"""

# Create the agent
print(f"Creating agent: {property_agent_name}")
property_agent = agents.create_agent(
    property_agent_name,
    agent_description,
    agent_instruction,
    agent_foundation_model
)
property_agent_id = property_agent[0]

# Define action group functions
functions_def = [
    {
        "name": "get_property_id_by_address",
        "description": "Finds property ID using physical address",
        "parameters": {
            "address": {
                "description": "Full street address including city and state",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_property_core",
        "description": "Retrieves core property details like price, size, and features",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_climate_info",
        "description": "Provides climate risk assessment for the property",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_neighborhood_info",
        "description": "Retrieves neighborhood details including schools and amenities",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_sales_tax_history",
        "description": "Provides sales history and tax assessment data",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_exterior_details",
        "description": "Retrieves exterior features and yard information",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_interior_details",
        "description": "Provides interior features and layout details",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_kitchen_details",
        "description": "Retrieves kitchen specifications and appliances",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_hoa_utilities",
        "description": "Provides HOA information and utility details",
        "parameters": {
            "property_id": {
                "description": "Unique property identifier", 
                "required": True,
                "type": "string"
            }
        }
    }
]


# Additional IAM policy for DynamoDB access
additional_iam_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
            "dynamodb:DeleteItem",
            "dynamodb:Query",
            "dynamodb:UpdateItem",
            "dynamodb:Scan"
        ],
        "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{property_table}*" # allow access to both primary and GSI
    }]
}

# Create Lambda function and add action group
print(f"Adding action group to agent with Lambda function: {property_lambda_name}")
agents.add_action_group_with_lambda(
    agent_name=property_agent_name,
    lambda_function_name=property_lambda_name,
    source_code_file="agents/property_search_info.py",
    agent_functions=functions_def,
    agent_action_group_name="property_actions",
    agent_action_group_description="Functions to access property data",
    additional_function_iam_policy=json.dumps(additional_iam_policy),
    dynamo_args=dynamoDB_args
)

# Create agent alias
print("Creating agent alias for testing")
property_agent_alias_id, property_agent_alias_arn = agents.create_agent_alias(
    property_agent_id, 'v1'
)

print(f"Agent ID: {property_agent_id}")
print(f"Agent Alias ID: {property_agent_alias_id}")
print(f"Agent Alias ARN: {property_agent_alias_arn}")
