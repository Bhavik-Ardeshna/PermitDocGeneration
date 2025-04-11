import logging

import numpy as np
import scipy.sparse as sp
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from pymilvus import model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.sparse import SpladeEmbeddingFunction
from sentence_transformers import SentenceTransformer

from constants import (
    DENSE_EMBEDDING_MODEL_PATH,
    SPARSE_EMBEDDING_MODEL_PATH,
    TOKENIZER_MODEL_PATH,
)
from src.db.utils import add_chunks_to_db, add_sam_chunks_to_db
from src.vec_db.utils import  insert_sam_chunks_to_milvus

# Initialize logger
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class Embeddings:
    """
    A class for generating dense and sparse embeddings using pre-trained models.

    Attributes:
        dense_ef (SentenceTransformerEmbeddingFunction): The dense embedding function using Sentence Transformer.
        sparse_ef (SpladeEmbeddingFunction): The sparse embedding function using Splade.
    """

    def __init__(self):
        self.dense_ef = SentenceTransformerEmbeddingFunction(
            model_name=DENSE_EMBEDDING_MODEL_PATH
        )
        self.sparse_ef = SpladeEmbeddingFunction(model_name=SPARSE_EMBEDDING_MODEL_PATH)
        self.token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=280, model_name=TOKENIZER_MODEL_PATH
        )

    def embed(self, df) -> list:
        """
        Embed the text using the Sentence Transformers model and Splade model.

        Args:
            df (pd.DataFrame): DataFrame containing the text data to be embedded.

        Returns:
            pd.DataFrame: DataFrame with added columns for dense and sparse embeddings.
        """
        # Convert the text to a list of text chunks
        text_list = list(df["text"].values)

        dense_embedded_text = self.dense_ef.encode_documents(text_list)
        sparse_embedded_text = self.sparse_ef.encode_documents(text_list)

        # Add columns to df
        df["dense_embedded_text"] = dense_embedded_text
        sparse_texts_list = [
            sparse_embedded_text[i] for i in range(sparse_embedded_text.shape[0])
        ]
        df["sparse_embedded_text"] = sparse_texts_list
        return df

    def encode_for_search(self, query: str) -> np.ndarray:
        """
        Encode the given text for search in the vector DB, with tokenization to improve accuracy.

        Args:
            search_text (str): The text input by the user to encode.

        Returns:
            np.ndarray: The normalized vector embedding of the tokenized input text.
        """
        # Tokenize the search text
        tokenized_texts = self.token_splitter.split_text(query)

        normalized_dense_encoded_texts = self.dense_ef.encode_queries(tokenized_texts)
        normalized_sparse_encoded_texts = self.sparse_ef.encode_queries(tokenized_texts)

        # Aggregate the embeddings if there are multiple, this step depends on your application's needs
        # For simplicity, we'll average the embeddings to get a single vector representation
        if len(normalized_dense_encoded_texts) > 1:
            aggregated_dense_embedding = [
                np.mean(normalized_dense_encoded_texts, axis=0)
            ]
        else:
            aggregated_dense_embedding = normalized_dense_encoded_texts

        if normalized_sparse_encoded_texts.shape[0] > 1:
            # Take mean of the sparse embeddings
            aggregated_sparse_embedding = sp.csr_matrix(
                np.mean(normalized_sparse_encoded_texts, axis=0)
            )
        else:
            aggregated_sparse_embedding = normalized_sparse_encoded_texts

        return aggregated_dense_embedding, aggregated_sparse_embedding



def embed_sam_and_send_to_db(df):
    """
    Splits the text into tokens, embeds it, and sends the embedded text to a Milvus database.
    Also, sends text chunks to a SQL database.

    Args:
        df (pd.DataFrame): DataFrame containing the text data to be embedded and sent to the databases.

    Raises:
        ValueError: If an error occurs during the process.
    """
    try:
        # Split text into tokens and embed it using a predefined embedding function.
        # `token_split_texts` is a list of text chunks, and `embeded_text` is the embedded representation of the text.
        df = Embeddings().embed(df=df)
        logger.info("Document has been embedded")

        # Send embedded text to Milvus database along with document ID and context.
        # `chunk_ids` is a list of IDs for the chunks stored in the Milvus database.
        chunk_ids = insert_sam_chunks_to_milvus(df)
        token_split_texts = list(df["text"].values)
        doc_id = df["doc_id"].iloc[0]

        # Send text chunks to a SQL database, linking them with their corresponding document ID and chunk IDs.
        add_sam_chunks_to_db(
            doc_id=doc_id, chunk_ids=chunk_ids, text_chunks=token_split_texts
        )

        logger.info("Text chunked and sent to databases.")

    except ValueError as ve:
        raise ValueError(ve)
