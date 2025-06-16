import boto3
from pprint import pprint

region = "us-east-1"
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name=region)

foundation_model_1 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
foundation_model = "amazon.nova-pro-v1:0"

response = bedrock_agent_runtime_client.retrieve_and_generate(
    input={
        "text": "Looking for 3 to 5 bedroom homes in Atlanta, Georgia built in 2025. Only return the properties that are matching user query with 2 to 3 sentences about each property.",
    },
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            'knowledgeBaseId': "IBTJXPOYRN",
            "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(region, foundation_model),
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults":10,
                    # "implicitFilterConfiguration": {
                    #     "metadataAttributes":[
                    #     {
                    #         "key": "year_built",
                    #         "type": "NUMBER",
                    #         "description": "The year the property was built."
                    #     },
                    #     {
                    #         "key": "city",
                    #         "type": "STRING",
                    #         "description": "The city name the property is located."
                    #     },
                    #     {
                    #         "key": "state",
                    #         "type": "STRING",
                    #         "description": "The state name the property is located."
                    #     },
                    #     {
                    #         "key": "address",
                    #         "type": "STRING",
                    #         "description": "The address of the property."
                    #     },
                    #     {
                    #         "key": "price",
                    #         "type": "NUMBER",
                    #         "description": "The price of the property."
                    #     },
                    #     {
                    #         "key": "square_footage",
                    #         "type": "NUMBER",
                    #         "description": "The square footage of the property."
                    #     },
                    #     {
                    #         "key": "property_id",
                    #         "type": "STRING",
                    #         "description": "The ID of the property."
                    #     }
                    # ],
                    # "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(region, foundation_model)
                    # }
                } 
            }
        }
    }
)

# print(response)
print(response['output']['text'],end='\n'*2)

# Loop through each citation
for i, citation in enumerate(response.get('citations', [])):
    retrieved_refs = citation.get('retrievedReferences', [])
    
    for ref in retrieved_refs:
        metadata = ref.get('metadata', {})
        for key, value in metadata.items():
            print(f"{key}: {value}")

# print("------------------------------")
# response = bedrock_agent_runtime_client.retrieve(
#     knowledgeBaseId="IBTJXPOYRN", 
#     nextToken='string',
#     retrievalConfiguration={
#         "vectorSearchConfiguration": {
#             "numberOfResults":10,
#             "implicitFilterConfiguration": {
#                     "metadataAttributes":[
#                         {
#                             "key": "year_built",
#                             "type": "NUMBER",
#                             "description": "The year the property was built."
#                         },
#                         {
#                             "key": "city",
#                             "type": "STRING",
#                             "description": "The city name the property is located."
#                         },
#                         {
#                             "key": "state",
#                             "type": "STRING",
#                             "description": "The state name the property is located."
#                         },
#                         {
#                             "key": "address",
#                             "type": "STRING",
#                             "description": "The address of the property."
#                         },
#                         {
#                             "key": "price",
#                             "type": "NUMBER",
#                             "description": "The price of the property."
#                         },
#                         {
#                             "key": "square_footage",
#                             "type": "NUMBER",
#                             "description": "The square footage of the property."
#                         },
#                         {
#                             "key": "property_id",
#                             "type": "STRING",
#                             "description": "The ID of the property."
#                         }
#                     ],
#                     "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(region, foundation_model_1)
#                 },
#         } 
#     },
#     retrievalQuery={
#         "text": "Looking for 3 to 5 bedroom homes in Georgetown, Texas built after 2020"
#     }
# )

# pprint(response)