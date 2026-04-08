# 🧠 RAG Knowledge Base Assistant

A production-grade Retrieval-Augmented Generation (RAG) chatbot for Data Engineering and AI topics. The system automatically refreshes its knowledge base daily, embeds content into a vector database, and serves answers through a Streamlit interface powered by Claude.

---

## Architecture

```
Data Sources                 Pipeline (Prefect)              Storage
─────────────────            ──────────────────────          ───────────────────
Wikipedia API   ──┐          fetch_wikipedia_articles        Pinecone (serverless)
                  ├──────►   ingest_azure_pdfs          ──►  384-dim cosine index
Azure Blob PDFs ──┘          chunk_documents (800 chars)
                             embed_and_sync (MiniLM)

Chat Application             Observability
─────────────────────────    ──────────────────────────
User → retrieve_context  ──► Snowflake: PIPELINE_RUN_LOG
     → generate_response     Snowflake: RAG_CHAT_HISTORY
     → Claude Sonnet stream
```

---

## Features

- **Daily pipeline** — Prefect flow runs at 6 AM (Mon–Fri), fetching 25 Wikipedia articles and 3 technical books from Azure Blob Storage
- **Semantic search** — `all-MiniLM-L6-v2` embeddings (384-dim) stored in Pinecone serverless
- **Streaming responses** — Claude Sonnet answers grounded strictly in retrieved context
- **Full observability** — every pipeline run and chat turn logged to Snowflake
- **Secure secrets** — all API keys fetched at runtime from Azure Key Vault

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Prefect |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Pinecone (serverless, AWS us-east-1) |
| LLM | Anthropic Claude Sonnet (`claude-sonnet-4-5`) |
| Frontend | Streamlit |
| PDF extraction | PyMuPDF (`fitz`) |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Secrets | Azure Key Vault + `DefaultAzureCredential` |
| Data storage | Azure Blob Storage |
| Logging | Snowflake |

---

## Knowledge Base

The assistant indexes the following content:

**Books (via Azure Blob Storage)**
- *Fundamentals of Data Engineering* — Reis & Housley
- *The Data Engineer's Guide to Apache Spark* — Databricks
- *Generative AI and LLMs for Dummies*

**Wikipedia articles (25 topics)**
Data engineering, ETL, Data pipeline, Snowflake, Apache Airflow, RAG, LLMs, Vector databases, Embeddings, Apache Spark, Apache Kafka, MLOps, Prompt engineering, Transformers, Semantic search, and more.

---

## Project Structure

```
.
├── app.py                  # Streamlit chat application
├── pipeline.py             # Prefect flow — ingestion, chunking, embedding
├── vector_store.py         # Pinecone client, upsert, and search utilities
├── snowflake_logger.py     # Pipeline run + chat turn logging to Snowflake
├── deployment.py           # Prefect deployment (cron schedule)
└── .env                    # Environment variables (see setup below)
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A Pinecone account (free tier works)
- An Anthropic API key
- Azure subscription (Key Vault + Blob Storage)
- Snowflake account
- Prefect account (or self-hosted server)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Key packages</summary>

```
streamlit
anthropic
sentence-transformers
pinecone-client
prefect
azure-storage-blob
azure-keyvault-secrets
azure-identity
pymupdf
langchain-text-splitters
snowflake-connector-python
wikipedia-api
python-dotenv
```
</details>

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
AZURE_KEY_VAULT_URL=https://<your-vault>.vault.azure.net/
AZURE_STORAGE_CONNECTION_STRING=<your-connection-string>
AZURE_CONTAINER_NAME=<your-container>
PINECONE_INDEX_NAME=<your-index-name>
```

### 4. Azure Key Vault secrets

Add the following secrets to your Key Vault:

| Secret name | Value |
|---|---|
| `ANTHROPIC-API-KEY` | Your Anthropic API key |
| `PINECONE-API-KEY` | Your Pinecone API key |
| `SNOWFLAKE-CONNECTION-STRING` | Snowflake connection string (`account=...;user=...;password=...`) |

### 5. Snowflake tables

```sql
CREATE TABLE IF NOT EXISTS RAG_ASSISTANT.PIPELINES.PIPELINE_RUN_LOG (
    run_id          VARCHAR        NOT NULL,       
    run_started_at  TIMESTAMP_NTZ  NOT NULL,
    run_finished_at TIMESTAMP_NTZ,
    status          VARCHAR(20),                    

    -- Ingestion counts
    wiki_articles_fetched   INT DEFAULT 0,
    pdfs_ingested           INT DEFAULT 0,
    total_chunks_created    INT DEFAULT 0,

    -- Vector sync stats
    vectors_upserted        INT DEFAULT 0,
    pinecone_index_name     VARCHAR,
    embedding_model         VARCHAR,

    -- Error tracking
    error_message           VARCHAR,
    duration_seconds        FLOAT,

    PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS RAG_ASSISTANT.PIPELINES.RAG_CHAT_HISTORY (
    message_id       VARCHAR       NOT NULL,
    session_id       VARCHAR       NOT NULL,
    asked_at         TIMESTAMP_NTZ NOT NULL,
    
    -- The conversation turn
    question         TEXT          NOT NULL,
    answer           TEXT          NOT NULL,

    -- RAG metadata
    sources_used     VARIANT,                     
    num_chunks_used  INT,
    top_source_title VARCHAR,

    -- Performance
    response_time_ms INT,

    PRIMARY KEY (message_id)
);
```

---

## Running the App

### Run the chat interface

```bash
streamlit run app.py
```

### Run the pipeline manually

```bash
python pipeline.py
```

### Deploy the scheduled pipeline

```bash
python deployment.py
```

This registers a Prefect deployment that runs the pipeline every weekday at 6 AM.

---

## How it works

1. **Ingestion** — The Prefect flow fetches Wikipedia articles and downloads PDFs from Azure Blob Storage.
2. **Chunking** — Documents are split into 800-character chunks with 80-character overlap using `RecursiveCharacterTextSplitter`.
3. **Embedding** — Each chunk is encoded with `all-MiniLM-L6-v2` (384 dimensions).
4. **Sync** — Vectors and metadata are upserted to Pinecone in batches of 100.
5. **Retrieval** — At query time, the user's question is embedded and the top-5 most similar chunks are retrieved.
6. **Generation** — Claude Sonnet is prompted with the retrieved context and streams the answer back to the UI.
7. **Logging** — Each Q&A turn (question, answer, sources, latency) is written to Snowflake.

---

## Configuration

| Parameter | Default | Location |
|---|---|---|
| Chunk size | 800 chars | `pipeline.py` |
| Chunk overlap | 80 chars | `pipeline.py` |
| Top-k retrieval | 5 chunks | `app.py` |
| Embedding model | `all-MiniLM-L6-v2` | `pipeline.py` / `app.py` |
| LLM | `claude-sonnet-4-5` | `app.py` |
| Max tokens | 1024 | `app.py` |
| Schedule | `0 6 * * 1-5` | `deployment.py` |

---

## Acknowledgements

- [Anthropic](https://www.anthropic.com) — Claude API
- [Pinecone](https://www.pinecone.io) — Vector database
- [Prefect](https://www.prefect.io) — Pipeline orchestration
- [Sentence Transformers](https://www.sbert.net) — `all-MiniLM-L6-v2`
- [Wikipedia-API](https://github.com/martin-majlis/Wikipedia-API) — Article fetching
