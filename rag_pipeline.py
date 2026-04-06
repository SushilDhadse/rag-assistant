import os
import chromadb
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load embedding model and ChromaDB
print("🔢 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("💾 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("rag_knowledge_base")
print(f"✅ Connected! Collection has {collection.count()} chunks\n")


def retrieve_context(question: str, n_results: int = 5) -> list:
    """Find most relevant chunks for a question."""

    # Embed the question
    question_embedding = model.encode(question).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    # Format results
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "url": meta["url"]
        })

    return chunks


def build_prompt(question: str, chunks: list) -> str:
    """Build the RAG prompt with context."""

    # Format context from retrieved chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['title']} ---\n"
        context += chunk["text"]
        context += "\n"

    # This is the key RAG prompt — telling Claude to use ONLY the context
    prompt = f"""You are a helpful Data Engineering and AI assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough 
information in my knowledge base to answer that."
Always mention which source(s) you used in your answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt


def ask(question: str) -> str:
    """Full RAG pipeline — retrieve, build prompt, ask Claude."""

    print(f"\n🔍 Searching knowledge base for: '{question}'")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_context(question)
    print(f"📚 Found {len(chunks)} relevant chunks from:")
    for c in chunks:
        print(f"   - {c['title']}")

    # Step 2: Build RAG prompt
    prompt = build_prompt(question, chunks)

    # Step 3: Ask Claude
    print("\n🤖 Asking Claude...\n")
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text
    return answer, chunks


if __name__ == "__main__":
    # Test with a few questions
    test_questions = [
        "What is a data pipeline?",
        "How does RAG work?",
        "What is the difference between a data lake and a data warehouse?",
    ]

    for question in test_questions:
        answer, chunks = ask(question)
        print(f"Answer: {answer}")
        print("\n" + "="*60 + "\n")