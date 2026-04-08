import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

def get_pinecone_client():
    """Fetch Pinecone API key from Azure Key Vault and create client."""
    key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
    api_key = secret_client.get_secret("PINECONE-API-KEY").value
    return Pinecone(api_key=api_key)

def get_or_create_index(pc: Pinecone, index_name: str):
    """Get existing index or create a new one."""
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"📦 Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,          # matches all-MiniLM-L6-v2 output   
            metric="cosine",        # same as ChromaDB setup
            spec=ServerlessSpec(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                cloud="aws",
                region="us-east-1"  # free tier region
            )
        )
        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("✅ Index ready!")
    else:
        print(f"✅ Using existing index: {index_name}")

    return pc.Index(index_name)

def upsert_chunks(index, chunks: list, embeddings: list):
    """Store chunks in Pinecone."""
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

    # Upsert in batches
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)

    return len(vectors)

def search(index, query_embedding: list, n_results: int = 5) -> list:
    """Search Pinecone for similar chunks."""
    results = index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True
    )

    chunks = []
    for match in results["matches"]:
        chunks.append({
            "text": match["metadata"]["text"],
            "title": match["metadata"]["title"],
            "url": match["metadata"]["url"],
            "score": match["score"]
        })

    return chunks