from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

# from constants import EMBEDDING_MODEL, TOKENIZER_MODEL


DENSE_EMBEDDING_MODEL = "Snowflake/arctic-embed-m"
TOKENIZER_MODEL = "Snowflake/arctic-embed-m"
SPARSE_EMBEDDING_MODEL = "naver/splade-cocondenser-ensembledistil"

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
print("Downloading model: ", DENSE_EMBEDDING_MODEL)
model_save_path = "/app/models/sentence_transformers/{}/".format(DENSE_EMBEDDING_MODEL)

model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
model.save(model_save_path)
print("Dense Model Downloaded")


if TOKENIZER_MODEL != DENSE_EMBEDDING_MODEL:

    # TOKENIZER_MODEL = "all-mpnet-base-v2"
    print("Downloading model: ", TOKENIZER_MODEL)
    model_save_path = "/app/models/sentence_transformers/{}/".format(TOKENIZER_MODEL)

    model = SentenceTransformer(TOKENIZER_MODEL)
    model.save(model_save_path)
else:
    print(
        "Tokenizer model is the same as the dense embedding model. Skipping download."
    )

if SPARSE_EMBEDDING_MODEL != DENSE_EMBEDDING_MODEL:
    print("Downloading Sparse Model", SPARSE_EMBEDDING_MODEL)
    sparse_model_save_path = "/app/models/sentence_transformers/{}/".format(
        SPARSE_EMBEDDING_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(SPARSE_EMBEDDING_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(SPARSE_EMBEDDING_MODEL)
    tokenizer.save_pretrained(sparse_model_save_path)
    model.save_pretrained(sparse_model_save_path)
    print("Sparse Model Downloaded")

else:
    print("Sparse model is the same as the dense model. Skipping download.")
