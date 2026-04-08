import os
import time
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pinecone_client(api_key: Optional[str] = None) -> Pinecone:
    """
    Initializes a Pinecone client.
    
    Args:
        api_key: Optional pre-fetched API key. If None, fetches from Azure Key Vault.
        
    Returns:
        Initialized Pinecone client instance.
    """
    if not api_key:
        logger.info("Retrieving Pinecone API key from Azure Key Vault.")
        try:
            key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
            credential = DefaultAzureCredential()
            secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
            api_key = secret_client.get_secret("PINECONE-API-KEY").value
        except Exception as e:
            logger.error(f"Failed to retrieve secret from Key Vault: {e}")
            raise

    return Pinecone(api_key=api_key)

def get_or_create_index(pc: Pinecone, index_name: str) -> Any:
    """
    Checks for index existence or initializes a new serverless index.
    
    Args:
        pc: Initialized Pinecone client.
        index_name: Target index name.
        
    Returns:
        Pinecone Index instance.
    """
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Initializing new Serverless Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,      # Optimized for sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
        # Poll for index readiness
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for index deployment...")
            time.sleep(1)
        logger.info("Index deployed successfully.")
    else:
        logger.info(f"Connected to existing index: {index_name}")

    return pc.Index(index_name)

def upsert_chunks(index: Any, chunks: List[Dict], embeddings: List[List[float]]) -> int:
    """
    Batches and uploads vector embeddings with associated metadata.
    
    Args:
        index: The Pinecone Index instance.
        chunks: List of dictionaries containing raw text and metadata.
        embeddings: List of calculated vector embeddings.
        
    Returns:
        Total number of vectors successfully upserted.
    """
    BATCH_SIZE = 100
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "title": chunk["title"],
                "url": chunk["url"],
                "chunk_index": chunk["chunk_index"],
                "source_type": chunk.get("source_type", "unknown")
            }
        })

    logger.info(f"Upserting {len(vectors)} vectors in batches of {BATCH_SIZE}.")
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)

    return len(vectors)

def search(index: Any, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
    """
    Performs semantic search retrieval and formats results.
    
    Args:
        index: The Pinecone Index instance.
        query_embedding: Vectorized user query.
        n_results: Top K results to return.
        
    Returns:
        Formatted list of matches including metadata and similarity scores.
    """
    results = index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True
    )

    return [
        {
            "text": match["metadata"]["text"],
            "title": match["metadata"]["title"],
            "url": match["metadata"]["url"],
            "score": round(match["score"], 4)
        }
        for match in results["matches"]
    ]