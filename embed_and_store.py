import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # progress bar

def load_chunks(path: str) -> list:
    """Load chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_and_store(chunks: list):
    """Embed chunks and store in ChromaDB."""

    # Load the free local embedding model
    print("🔢 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded!\n")

    # Set up ChromaDB — stores data locally in ./chroma_db folder
    print("💾 Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete collection if it already exists (clean start)
    try:
        client.delete_collection("rag_knowledge_base")
        print("🗑️  Cleared existing collection")
    except:
        pass

    # Create a fresh collection
    collection = client.create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    print("✅ Collection created!\n")

    # Process chunks in batches of 100
    BATCH_SIZE = 100
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"🚀 Embedding and storing {len(chunks)} chunks in {total_batches} batches...\n")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]

        # Extract texts to embed
        texts = [c["text"] for c in batch]

        # Generate embeddings
        embeddings = model.encode(texts).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "title": c["title"],
                "url": c["url"],
                "chunk_index": c["chunk_index"]
            } for c in batch]
        )

    print(f"\n🎉 Done! Stored {collection.count()} chunks in ChromaDB")
    return collection

def test_search(collection, model):
    """Quick test to verify search is working."""
    print("\n🔍 Testing semantic search...")

    test_query = "What is a data pipeline?"
    query_embedding = model.encode(test_query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\nQuery: '{test_query}'")
    print("\nTop 3 results:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {meta['title']}")
        print(f"Text: {doc[:200]}...")

if __name__ == "__main__":
    # Load chunks
    print("📂 Loading chunks...")
    chunks = load_chunks("data/chunks.json")
    print(f"✅ Loaded {len(chunks)} chunks\n")

    # Embed and store
    collection = embed_and_store(chunks)

    # Load model again for testing
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Test it works
    test_search(collection, model)