import os
import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

import fitz
import wikipediaapi
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Internal utilities
from vector_store import get_pinecone_client, get_or_create_index, upsert_chunks
from snowflake_logger import PipelineRunLogger

load_dotenv()

# --- Configuration Constants ---
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

def clean_text(text: str) -> str:
    """
    Standardizes and cleans extracted text for embedding.
    
    Args:
        text: Raw text string from source.
        
    Returns:
        Sanitized text string.
    """
    # Remove Table of Contents dots
    text = re.sub(r'\.{3,}', ' ', text)
    # Remove non-ASCII characters (encoding artifacts)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Normalize whitespace and remove newlines/tabs
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

@task(name="Fetch Wikipedia Articles", retries=2)
def fetch_wikipedia_articles() -> List[Dict[str, Any]]:
    """
    Retrieves content from Wikipedia for defined topics.
    """
    logger = get_run_logger()
    wiki = wikipediaapi.Wikipedia(
        language="en", 
        user_agent="DataEngineeringAssistant/1.0 (contact: sushildhadse@example.com)"
    )
    
    articles = []
    for topic in TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                articles.append({
                    "title": page.title, 
                    "url": page.fullurl, 
                    "text": page.text, 
                    "source_type": "wikipedia"
                })
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia topic '{topic}': {e}")
            
    logger.info(f"Successfully fetched {len(articles)} Wikipedia articles.")
    return articles

@task(name="Ingest Azure PDF Documents")
def ingest_azure_pdfs() -> List[Dict[str, Any]]:
    """
    Downloads and extracts text from PDF blobs in Azure Storage.
    """
    logger = get_run_logger()
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_CONTAINER_NAME")
    
    if not connect_str or not container:
        raise ValueError("Azure Storage credentials missing from environment variables.")
        
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    book_data = []
    
    for book in BOOKS:
        try:
            logger.info(f"Processing book: {book['title']}")
            blob_client = blob_service_client.get_blob_client(
                container=container, 
                blob=book["blob_name"]
            )
            pdf_bytes = blob_client.download_blob().readall()
            
            # Stream PDF content into text
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_book_text = "".join([page.get_text() for page in doc])
            
            book_data.append({
                "title": book["title"],
                "url": book["url"],
                "text": clean_text(full_book_text),
                "source_type": "book"
            })
        except Exception as e:
            logger.error(f"Failed to process book '{book['title']}': {e}")
            
    return book_data

@task(name="Document Chunking")
def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Segments long documents into chunks for semantic indexing.
    """
    logger = get_run_logger()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['source_type']}_{doc['title']}_{i}",
                "title": doc["title"],
                "url": doc["url"],
                "text": chunk_text,
                "source_type": doc["source_type"],
                "chunk_index": i
            })
            
    logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents.")
    return all_chunks

@task(name="Pinecone Synchronization")
def embed_and_sync(chunks: List[Dict[str, Any]]) -> int:
    """
    Generates embeddings and synchronizes chunks with Pinecone Vector DB.
    """
    logger = get_run_logger()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialization using shared utility
    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = get_or_create_index(pc, index_name)
    
    texts = [c["text"] for c in chunks]
    logger.info("Generating embeddings for all chunks...")
    embeddings = model.encode(texts).tolist()
    
    count = upsert_chunks(index, chunks, embeddings)
    logger.info(f"Successfully synchronized {count} vectors with Pinecone.")
    return count

@flow(name="RAG Knowledge Base Refresh")
def rag_pipeline():
    """
    Main orchestration flow for the Knowledge Base refresh.
    """
    logger = get_run_logger()
    sf_logger  = PipelineRunLogger()

    try:
        logger.info("Starting Knowledge Base Refresh Pipeline.")

        # Ingestion Layer
        wiki_docs = fetch_wikipedia_articles()
        pdf_docs = ingest_azure_pdfs()

        sf_logger.update(
                wiki_articles_fetched=len(wiki_docs),
                pdfs_ingested=len(pdf_docs),
            )
        
        # Transformation Layer
        all_chunks = chunk_documents(wiki_docs + pdf_docs)
        sf_logger.update(total_chunks_created=len(all_chunks))
        
        # Loading Layer
        count = embed_and_sync(all_chunks)
        sf_logger.update(vectors_upserted=count)
        
        logger.info(f"Pipeline execution completed. Total vectors processed: {count}")
        sf_logger.commit(status="SUCCESS")
    
    except Exception as e:
        sf_logger.error_message = str(e)
        sf_logger.commit(status="FAILED")
        raise

if __name__ == "__main__":
    rag_pipeline()