import os
import streamlit as st
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from vector_store import get_pinecone_client, get_or_create_index, search

# Load Environment Variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# --- Resource Loading (Cached) ---
@st.cache_resource
def initialize_services():
    """Single point of entry for all external services and models."""
    with st.spinner("🔢 Connecting to services..."):
        # 1. Fetch all Secrets from Azure Key Vault
        kv_url = os.getenv("AZURE_KEY_VAULT_URL")
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=kv_url, credential=credential)
        
        anthropic_key = secret_client.get_secret("ANTHROPIC-API-KEY").value
        pinecone_key = secret_client.get_secret("PINECONE-API-KEY").value
        
        # 2. Initialize Clients
        ai_client = anthropic.Anthropic(api_key=anthropic_key)
        pc = get_pinecone_client() 
        
        # 3. Load Index and Embedding Model
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = get_or_create_index(pc, index_name)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
    return ai_client, model, index

# Global instances
client, model, index = initialize_services()

# --- Page Config ---
st.set_page_config(
    page_title="Data Engineering Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- RAG Functions ---
def retrieve_context(question: str, n_results: int = 5) -> list:
    """Find most relevant chunks using Pinecone."""
    question_embedding = model.encode(question).tolist()
    return search(index, question_embedding, n_results)

def ask_with_streaming(question: str):
    """Orchestrates retrieval and streaming response."""
    # Step 1: Retrieve chunks
    chunks = retrieve_context(question)
    
    # Step 2: Format context for prompt
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"\n--- Source {i+1}: {chunk['title']} ---\n{chunk['text']}\n"

    # Step 3: Stream from Claude
    full_reply = ""
    placeholder = st.empty()
    
    system_instr = (
        "You are a helpful Data Engineering and AI assistant. "
        "Answer the user's question using ONLY the context provided. "
        "If the answer is not in the context, say 'I don't have enough "
        "information in my knowledge base to answer that.' "
        "Be concise and clear. Use bullet points where helpful."
    )

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_instr,
        messages=[{"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"}]
    ) as stream:
        for text in stream.text_stream:
            full_reply += text
            placeholder.markdown(full_reply + "▌")

    placeholder.markdown(full_reply)
    return full_reply, chunks

# --- UI Layout ---
st.title("🧠 Data Engineering Knowledge Assistant")
st.caption("📊 Vector DB: Pinecone ☁️ | Powered by Claude AI + RAG")

# Sidebar
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.markdown("""
    **📖 Books**
    - Fundamentals of Data Engineering
    - The Data Engineer's Guide to Apache Spark
    - Generative AI & LLMs for Dummies
    
    **🌐 Wikipedia Articles**
    - Data Pipelines, Snowflake, Spark, Kafka, MLOps, and more.
    """)
    st.divider()
    
    # Display Index Stats
    stats = index.describe_index_stats()
    st.caption(f"📊 {stats['total_vector_count']} chunks indexed")
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources used"):
                seen = {chunk["title"] for chunk in message["sources"]}
                for title in seen:
                    # Find first matching URL for the title
                    url = next(c["url"] for c in message["sources"] if c["title"] == title)
                    st.markdown(f"- [{title}]({url})")

# User Input Logic
if prompt := st.chat_input("Ask me anything about Data Engineering or AI..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            answer, chunks = ask_with_streaming(prompt)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })
    st.rerun()