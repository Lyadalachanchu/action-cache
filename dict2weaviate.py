import weaviate
import os
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wc
import weaviate.classes.init as wvc
from dotenv import load_dotenv
load_dotenv()

# Try to connect to Weaviate Cloud first, fallback to local
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

if weaviate_url and weaviate_api_key:
    print(f"Connecting to Weaviate Cloud: {weaviate_url}")
    try:
        # Configure additional settings to handle gRPC issues
        additional_config = wvc.AdditionalConfig(
            timeout=wvc.Timeout(init=30)  # Increase timeout
        )
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url, 
            auth_credentials=AuthApiKey(weaviate_api_key),
            additional_config=additional_config,
            skip_init_checks=True  # Skip gRPC health checks
        )
        print("Successfully connected to Weaviate Cloud!")
    except Exception as e:
        print(f"Failed to connect to Weaviate Cloud: {e}")
        print("Falling back to local connection...")
        try:
            client = weaviate.connect_to_local()
            print("Successfully connected to local Weaviate!")
        except Exception as local_e:
            print(f"Failed to connect to local Weaviate: {local_e}")
            print("Please ensure Weaviate is running locally or check your cloud credentials.")
            exit(1)
else:
    print("No Weaviate Cloud credentials found. Trying local connection...")
    try:
        client = weaviate.connect_to_local()
        print("Successfully connected to local Weaviate!")
    except Exception as e:
        print(f"Failed to connect to local Weaviate: {e}")
        print("Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables or start a local Weaviate instance.")
        exit(1)

client.collections.create(
    name="MovieCustomVector",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="overview", data_type=wc.DataType.TEXT),
        wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
        wc.Property(name="release_date", data_type=wc.DataType.DATE),
        wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
    ],
    # Define the vectorizer module (none, as we will add our own vectors)
    vector_config=wc.Configure.Vectors.self_provided(),
    # Define the generative module
    generative_config=wc.Configure.Generative.cohere()
)

client.close()