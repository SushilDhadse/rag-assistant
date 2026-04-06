import os
import streamlit as st
import chromadb
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Page Config ---
st.set_page_config(
    page_title="Data Engineering Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Load Models (cached so they don't reload on every message) ---
@st.cache_resource
def load_models():
    """Load embedding model and ChromaDB — cached after first load."""
    with st.spinner("🔢 Loading embedding model... please wait"):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("rag_knowledge_base")
    return model, collection

model, collection = load_models()

# --- RAG Functions ---
def retrieve_context(question: str, n_results: int = 5) -> list:
    """Find most relevant chunks for a question."""
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
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
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['title']} ---\n"
        context += chunk["text"]
        context += "\n"

    return f"""You are a helpful Data Engineering and AI assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough 
information in my knowledge base to answer that."
Be concise and clear. Use bullet points where helpful.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

def ask_with_streaming(question: str):
    """Full RAG pipeline with streaming."""

    # Retrieve relevant chunks
    chunks = retrieve_context(question)

    # Build prompt
    prompt = build_prompt(question, chunks)

    # Stream response from Claude
    full_reply = ""
    placeholder = st.empty()

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            full_reply += text
            placeholder.markdown(full_reply + "▌")

    placeholder.markdown(full_reply)

    return full_reply, chunks

# --- UI ---
st.title("🧠 Data Engineering Knowledge Assistant")
st.caption("Powered by Claude AI + RAG | Knowledge base: 25 Wikipedia articles")

# Sidebar with info
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.markdown("""
    This assistant knows about:
    - Data Pipelines & ETL
    - Snowflake & Data Warehouses
    - Apache Spark & Kafka
    - RAG & LLMs
    - Vector Databases
    - MLOps & Prompt Engineering
    - And much more!
    """)
    st.divider()
    st.caption(f"📊 {collection.count()} chunks indexed")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources used"):
                seen = set()
                for chunk in message["sources"]:
                    if chunk["title"] not in seen:
                        seen.add(chunk["title"])
                        st.markdown(f"- [{chunk['title']}]({chunk['url']})")

# Handle new user input
if prompt := st.chat_input("Ask me anything about Data Engineering or AI..."):

    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            answer, chunks = ask_with_streaming(prompt)

    # Save assistant message with sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })
    st.rerun()