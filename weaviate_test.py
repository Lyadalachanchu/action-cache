import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
import os
from dotenv import load_dotenv

load_dotenv()

# Get credentials from environment
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

# Connect with increased timeout (10 seconds is usually enough)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    additional_config=AdditionalConfig(timeout=Timeout(init=10))
)

print(f"Connected: {client.is_ready()}")
client.close()