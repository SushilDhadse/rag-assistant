import os
import json
import re
import fitz
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from prefect import flow, task, get_run_logger
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import get_pinecone_client, get_or_create_index, upsert_chunks
import wikipediaapi
from tqdm import tqdm

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

TOPICS = [
    "Data engineering", "Extract, transform, load", "Data pipeline",
    "Snowflake Inc", "Apache Airflow", "Retrieval-augmented generation",
    "Large language model", "Vector database", "Embedding (machine learning)",
    "Python (programming language)", "Data warehouse", "Data lake",
    "Apache Spark", "Apache Kafka", "Distributed computing", "MLOps",
    "Prompt engineering", "Transformer (deep learning)", "Semantic search",
    "Knowledge graph", "CI/CD", "Data governance", "Data mesh",
    "Tokenization", "Graph database"
]

BOOKS = [
    {
        "blob_name": "fundamentals_of_data_engineering.pdf.pdf",
        "title": "Fundamentals of Data Engineering",
        "url": "https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/"
    },
    {
        "blob_name": "The-Data-Engineers-Guide-to-Apache-Spark.pdf",
        "title": "The Data Engineer's Guide to Apache Spark",
        "url": "https://www.databricks.com/resources/ebook/learning-spark-lightning-fast-data-analytics"
    },
    {
        "blob_name": "Generative-AI-and-LLMs-for-Dummies.pdf",
        "title": "Generative AI and LLMs for Dummies",
        "url": "https://www.dummies.com/article/technology/information-technology/ai/generative-ai/"
    }
]

# -------------------------------------------------------
# TASK 1: Fetch Wikipedia Articles
# -------------------------------------------------------
@task(name="Fetch Wikipedia Articles", retries=2, retry_delay_seconds=30)
def fetch_wikipedia_articles() -> list:
    logger = get_run_logger()
    logger.info(f"Fetching {len(TOPICS)} Wikipedia articles...")

    wiki = wikipediaapi.Wikipedia(language="en", user_agent="rag-assistant/1.0")
    articles = []

    for topic in TOPICS:
        page = wiki.page(topic)
        if page.exists():
            articles.append({
                "title": page.title,
                "url": page.fullurl,
                "text": page.text
            })
            logger.info(f"✅ Fetched: {page.title}")
        else:
            logger.warning(f"❌ Not found: {topic}")

    logger.info(f"Fetched {len(articles)} articles")
    return articles


# -------------------------------------------------------
# TASK 2: Download PDFs from Azure
# -------------------------------------------------------
@task(name="Download PDFs from Azure", retries=2, retry_delay_seconds=30)
def download_pdfs_from_azure() -> list:
    logger = get_run_logger()
    logger.info("Connecting to Azure Blob Storage...")

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    all_pages = []

    for book in BOOKS:
        logger.info(f"Downloading: {book['title']}")

        # Download PDF
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME,
            blob=book["blob_name"]
        )
        pdf_bytes = blob_client.download_blob().readall()

        # Extract and clean text
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            text = clean_text(text)
            if len(text) < 100:
                continue
            all_pages.append({
                "title": book["title"],
                "url": book["url"],
                "page": i + 1,
                "text": text
            })

        logger.info(f"✅ Extracted pages from {book['title']}")

    logger.info(f"Total pages extracted: {len(all_pages)}")
    return all_pages


# -------------------------------------------------------
# TASK 3: Chunk All Documents
# -------------------------------------------------------
@task(name="Chunk Documents")
def chunk_documents(articles: list, book_pages: list) -> list:
    logger = get_run_logger()
    logger.info("Chunking all documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    all_chunks = []

    # Chunk Wikipedia articles
    for article in articles:
        chunks = splitter.split_text(article["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"wiki_{article['title']}_{i}",
                "title": article["title"],
                "url": article["url"],
                "chunk_index": i,
                "source_type": "wikipedia",
                "text": chunk
            })

    wiki_count = len(all_chunks)
    logger.info(f"Wikipedia chunks: {wiki_count}")

    # Chunk book pages
    for page in book_pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"book_{page['title']}_p{page['page']}_{i}",
                "title": page["title"],
                "url": page["url"],
                "chunk_index": i,
                "source_type": "book",
                "text": chunk
            })

    book_count = len(all_chunks) - wiki_count
    logger.info(f"Book chunks: {book_count}")
    logger.info(f"Total chunks: {len(all_chunks)}")

    return all_chunks


# -------------------------------------------------------
# TASK 4: Embed and Store in Pinecone
# -------------------------------------------------------
@task(name="Embed and Store in Pinecone")
def embed_and_store(chunks: list) -> int:
    logger = get_run_logger()
    logger.info("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Connecting to Pinecone via Azure Key Vault...")
    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = get_or_create_index(pc, index_name)

    logger.info(f"Embedding {len(chunks)} chunks...")

    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts).tolist()
        all_embeddings.extend(embeddings)

    logger.info("Storing in Pinecone...")
    count = upsert_chunks(index, chunks, all_embeddings)

    logger.info(f"✅ Stored {count} chunks in Pinecone")
    return count


# -------------------------------------------------------
# TASK 5: Notify Completion
# -------------------------------------------------------
@task(name="Notify Completion")
def notify_completion(chunk_count: int):
    logger = get_run_logger()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 50)
    logger.info("🎉 RAG Knowledge Base Refresh Complete!")
    logger.info(f"⏰ Timestamp: {timestamp}")
    logger.info(f"📊 Total chunks indexed: {chunk_count}")
    logger.info("=" * 50)


# -------------------------------------------------------
# HELPER
# -------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'CHAPTER\s+\d+.*?\n', '', text)
    text = re.sub(r'These materials are ©.*?\n', '', text)
    text = re.sub(r'.*?strictly prohibited.*?\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# -------------------------------------------------------
# FLOW
# -------------------------------------------------------
@flow(name="RAG Knowledge Base Refresh")
def rag_refresh_pipeline():
    logger = get_run_logger()
    logger.info("🚀 Starting RAG Knowledge Base Refresh Pipeline")

    # Run all tasks
    articles = fetch_wikipedia_articles()
    book_pages = download_pdfs_from_azure()
    chunks = chunk_documents(articles, book_pages)
    chunk_count = embed_and_store(chunks)
    notify_completion(chunk_count)


if __name__ == "__main__":
    rag_refresh_pipeline()
