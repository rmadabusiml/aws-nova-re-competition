from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

# client = OpenSearch(
#     hosts=[{'host': '1hu0dtumr17dixj14ht6.us-east-1.aoss.amazonaws.com', 'port': 443}],
#     use_ssl=True,
#     verify_certs=True,
#     connection_class=RequestsHttpConnection
# )

host = '1hu0dtumr17dixj14ht6.us-east-1.aoss.amazonaws.com'
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, 'us-east-1', 'aoss')

client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )

index_body = {
    "settings": {
        "index.knn": True
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "engine": "nmslib"
                }
            },
            "s3_key": {"type": "keyword"},
            "description": {"type": "text"}
        }
    }
}

client.indices.create(index="property-search-images", body=index_body)
