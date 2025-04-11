import os

from dotenv import load_dotenv
from google.cloud import secretmanager

load_dotenv()

# Initialize google cloud secret manager
client = secretmanager.SecretManagerServiceClient()

PROJECT_ID = os.getenv("PROJECT_ID")

LOCATION = "us"  # Format is 'us' or 'eu'


DB_USER = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_POSTGRES_USER/versions/latest"
).payload.data.decode("utf-8")
os.environ["DB_USER"] = DB_USER
DB_PASS = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_POSTGRES_PASSWORD/versions/latest"
).payload.data.decode("utf-8")
os.environ["DB_PASS"] = DB_PASS
DB_NAME = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_POSTGRES_DB/versions/latest"
).payload.data.decode("utf-8")
os.environ["DB_NAME"] = DB_NAME
DB_HOST = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/POSTGRES_SERVER/versions/latest"
).payload.data.decode("utf-8")
os.environ["DB_HOST"] = DB_HOST
DB_PORT = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/POSTGRES_PORT/versions/latest"
).payload.data.decode("utf-8")
os.environ["DB_PORT"] = DB_PORT

MILVUS_URI = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/GCP_MILVUS_URI/versions/latest"
).payload.data.decode("utf-8")
os.environ["MILVUS_URI"] = MILVUS_URI
COLLECTION_NAME_REGULATIONS = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/HYBRID_SRCH_COLLCTN_NAME_REGULATION/versions/latest"
).payload.data.decode("utf-8")
os.environ["COLLECTION_NAME_REGULATIONS"] = COLLECTION_NAME_REGULATIONS
COLLECTION_NAME_SAM = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/HYBRID_SRCH_COLLCTN_NAME_SAM/versions/latest"
).payload.data.decode("utf-8")
os.environ["COLLECTION_NAME_SAM"] = COLLECTION_NAME_SAM
TOKENIZERS_PARALLELISM = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/UNSTCTRD_TOKENIZERS_PARALLELISM/versions/latest"
).payload.data.decode("utf-8")
os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
ATLAS_JWT_SECRET_KEY = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_JWT_SECRET_KEY/versions/latest"
).payload.data.decode("utf-8")
os.environ["ATLAS_JWT_SECRET_KEY"] = ATLAS_JWT_SECRET_KEY
BACKEND_URI = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/UNSTCTRD_TO_BKND_URL/versions/latest"
).payload.data.decode("utf-8")
os.environ["BACKEND_URI"] = BACKEND_URI
GCP_SECRET_KEY = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/GCP_AUTH_TOKEN_BACKEND/versions/latest"
).payload.data.decode("utf-8")
os.environ["GCP_SECRET_KEY"] = GCP_SECRET_KEY
JWT_SETTINGS = eval(
    client.access_secret_version(
        name=f"projects/{PROJECT_ID}/secrets/GCP_AUTH_TOKEN_BACKEND/versions/latest"
    ).payload.data.decode("utf-8")
)
INFERENCE_LLM_BASE_URL = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_INFERENCE_URL/versions/latest"
).payload.data.decode("utf-8")
os.environ["INFERENCE_LLM_BASE_URL"] = INFERENCE_LLM_BASE_URL

MODEL_NAME = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_MODEL_NAME/versions/latest"
).payload.data.decode("utf-8")
os.environ["MODEL_NAME"] = MODEL_NAME

TOGETHER_API_KEY = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/TOGETHER_API_KEY/versions/latest"
).payload.data.decode("utf-8")
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

OPENAI_API_KEY = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/OPENAI_API_KEY/versions/latest"
).payload.data.decode("utf-8")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

VECTOR_DIMENSIONS = 768

DENSE_EMBEDDING_MODEL = "Snowflake/arctic-embed-m"
SPARSE_EMBEDDING_MODEL = "naver/splade-cocondenser-ensembledistil"
TOKENIZER_MODEL = "Snowflake/arctic-embed-m"

DENSE_EMBEDDING_MODEL_PATH = "/app/models/sentence_transformers/{}/".format(
    DENSE_EMBEDDING_MODEL
)
TOKENIZER_MODEL_PATH = "/app/models/sentence_transformers/{}/".format(TOKENIZER_MODEL)
SPARSE_EMBEDDING_MODEL_PATH = "/app/models/sentence_transformers/{}/".format(
    SPARSE_EMBEDDING_MODEL
)

# DENSE_EMBEDDING_MODEL_PATH = "Snowflake/arctic-embed-m"
# SPARSE_EMBEDDING_MODEL_PATH = "naver/splade-cocondenser-ensembledistil"
# TOKENIZER_MODEL_PATH = "Snowflake/arctic-embed-m"


CHUNK_OVERLAP = 0
TOKENS_PER_CHUNK = 280  # 20 tokens reserved for page number metadata
COLLECTION_NAME_SAM = os.getenv("COLLECTION_NAME_SAM")
COLLECTION_NAME_REGULATIONS = os.getenv("COLLECTION_NAME_REGULATIONS")

DETECTION_CLASS_PROB_THRESHOLD = 0.3

BACKEND_URI = os.getenv("BACKEND_URI")


# Get logger level and name of logger
logger_level = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_LOGGER_LEVEL/versions/latest"
).payload.data.decode("utf-8")
logger_name = client.access_secret_version(
    name=f"projects/{PROJECT_ID}/secrets/ATLAS_LOGGER_NAME/versions/latest"
).payload.data.decode("utf-8")

# Set into env variables so that loggers.py can access
os.environ["ATLAS_LOGGER_LEVEL"] = logger_level
os.environ["ATLAS_LOGGER_NAME"] = logger_name
