import os
import time
import uuid
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Internal utility imports
from vector_store import get_pinecone_client, get_or_create_index, search
from snowflake_logger import log_chat_turn

# Load configuration
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Standard logging for monitoring application performance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Resource Loading (Cached) ---
@st.cache_resource
def initialize_services() -> Tuple[anthropic.Anthropic, SentenceTransformer, Any]:
    """
    Initializes cloud clients and local ML models in a single cached call.
    Retrieves all sensitive credentials from Azure Key Vault.
    """
    # 1. Access Azure Key Vault
    kv_url = os.getenv("AZURE_KEY_VAULT_URL")
    if not kv_url:
        st.error("Azure Key Vault URL missing from environment configuration.")
        st.stop()
        
    try:
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=kv_url, credential=credential)
        
        # 2. Fetch API Secrets
        anthropic_key = secret_client.get_secret("ANTHROPIC-API-KEY").value
        pinecone_key = secret_client.get_secret("PINECONE-API-KEY").value
        
        # 3. Initialize Production Clients
        ai_client = anthropic.Anthropic(api_key=anthropic_key)
        pc_client = get_pinecone_client()
        
        # 4. Load Index and Embedding Model (384-dimensional)
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = get_or_create_index(pc_client, index_name)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        return ai_client, model, index
        
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        st.error("Failed to connect to backend services. Check logs for details.")
        st.stop()

# Shared singleton instances
client, model, index = initialize_services()

# --- Application Configuration ---
st.set_page_config(
    page_title="Data Engineering Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Logic Layer ---
def retrieve_context(question: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs semantic retrieval to find relevant knowledge chunks.
    """
    question_embedding = model.encode(question).tolist()
    return search(index, question_embedding, n_results)

def generate_rag_response(question: str):
    """
    Orchestrates the RAG flow: Retrieval -> Prompt Construction -> Streaming Generation.
    """
    # Step 1: Context Retrieval
    chunks = retrieve_context(question)
    
    # Step 2: Context Formatting
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"\n--- Source {i+1}: {chunk['title']} ---\n{chunk['text']}\n"

    # Step 3: LLM Interaction with System Framing
    full_reply = ""
    placeholder = st.empty()
    
    system_instruction = (
        "You are an expert Data Engineering and AI assistant. "
        "Answer the user's question using ONLY the context provided. "
        "If the answer is not in the context, state that you do not have "
        "enough information in your knowledge base. "
        "Maintain a professional, concise tone and use markdown for clarity."
    )

    try:
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system_instruction,
            messages=[{"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"}]
        ) as stream:
            for text in stream.text_stream:
                full_reply += text
                placeholder.markdown(full_reply + "▌")
    except Exception as e:
        logger.error(f"LLM Generation Error: {e}")
        full_reply = "I encountered an error generating a response. Please try again."

    placeholder.markdown(full_reply)
    return full_reply, chunks

# --- UI Layout ---
st.title("🧠 Data Engineering Knowledge Assistant")
st.caption("Vector DB: Pinecone (Serverless) | Engine: Claude Sonnet 4.5 | Orchestration: Prefect")

# Sidebar Implementation
with st.sidebar:
    st.header("Knowledge Base")
    st.markdown("""
    **Indexed Content**
    - Fundamentals of Data Engineering
    - The Data Engineer's Guide to Apache Spark
    - Generative AI & LLMs for Dummies
    - Curated Wikipedia articles (ETL, MLOps, Snowflake, etc.)
    """)
    st.divider()
    
    try:
        stats = index.describe_index_stats()
        st.metric("Indexed Chunks", stats['total_vector_count'])
    except Exception:
        st.caption("Status: Knowledge base stats temporarily unavailable.")
    
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Conversation Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Reference Sources"):
                unique_sources = {chunk["title"]: chunk["url"] for chunk in message["sources"]}
                for title, url in unique_sources.items():
                    st.markdown(f"- [{title}]({url})")

# User Query Interaction
if prompt := st.chat_input("Query the Data Engineering knowledge base..."):
    # Record and display user intent
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            t0 = time.time()
            answer, chunks = generate_rag_response(prompt)
            elapsed_ms = int((time.time() - t0) * 1000)

            log_chat_turn(
                session_id       = st.session_state.session_id,
                question         = prompt,
                answer           = answer,
                chunks           = chunks,
                response_time_ms = elapsed_ms,
            )

    # Persist assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })
    st.rerun()