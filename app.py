import os
import streamlit as st
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from sentence_transformers import SentenceTransformer
from vector_store import get_pinecone_client, get_or_create_index, search

# Load API key
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


# Fetch API key from Azure Key Vault
def get_anthropic_client():
    """Fetch API key from Azure Key Vault and create Anthropic client."""
    key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
    api_key = secret_client.get_secret("ANTHROPIC-API-KEY").value
    return anthropic.Anthropic(api_key=api_key)

client = get_anthropic_client()

# --- Page Config ---
st.set_page_config(
    page_title="Data Engineering Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    """Load embedding model and Pinecone index."""
    with st.spinner("🔢 Loading models..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        pc = get_pinecone_client()
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = get_or_create_index(pc, index_name)
    return model, index

model, index = load_models()

# --- RAG Functions ---
def retrieve_context(question: str, n_results: int = 5) -> list:
    """Find most relevant chunks for a question."""
    question_embedding = model.encode(question).tolist()
    return search(index, question_embedding, n_results)

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
st.caption(f"📊 Vector DB: Pinecone ☁️")
st.caption("Powered by Claude AI + RAG | Knowledge base: 25 Wikipedia articles")

# Sidebar with info
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.markdown("""
    This assistant knows about:
    
    **📖 Books**
    - Fundamentals of Data Engineering
    - The Data Engineer's Guide to Apache Spark
    - Generative AI & LLMs for Dummies
    
    **🌐 Wikipedia Articles**
    - Data Pipelines & ETL
    - Snowflake & Data Warehouses
    - Apache Spark & Kafka
    - RAG & LLMs
    - Vector Databases
    - MLOps & Prompt Engineering
    - And much more!
    """)
    st.divider()
    stats = index.describe_index_stats()
    st.caption(f"📊 {stats['total_vector_count']} chunks indexed")
    st.caption(f"📚 3 books + 25 Wikipedia articles")
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