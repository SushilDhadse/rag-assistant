import os
import json
import fitz  # pymupdf
from pathlib import Path
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import re

# Load env variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Books and their metadata
BOOKS = [
    {
        "blob_name": "fundamentals_of_data_engineering.pdf.pdf",
        "title": "Fundamentals of Data Engineering",
        "url": "https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/"
    },
    {
        "blob_name": "The-Data-Engineers-Guide-to-Apache-Spark.pdf",
        "title": "The Data Engineer's Guide to Apache Spark",
        "url": "https://pages.databricks.com/rs/094-YMS-629/images/LearningSpark_2.0.pdf"
    },
    {
        "blob_name": "Generative-AI-and-LLMs-for-Dummies.pdf",
        "title": "Generative AI and LLMs for Dummies",
        "url": "https://www.dummies.com/article/technology/information-technology/ai/generative-ai/"
    }
]

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""

    # Remove page numbers (standalone numbers)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove chapter headers like "CHAPTER 2  Understanding Large Language Models  11"
    text = re.sub(r'CHAPTER\s+\d+.*?\n', '', text)

    # Remove copyright lines
    text = re.sub(r'These materials are ©.*?\n', '', text)
    text = re.sub(r'.*?strictly prohibited.*?\n', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

def download_blob(blob_service_client, blob_name: str) -> bytes:
    """Download a blob as bytes."""
    print(f"  ⬇️  Downloading {blob_name}...")
    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )
    return blob_client.download_blob().readall()

def extract_text_from_pdf(pdf_bytes: bytes, title: str, url: str) -> list:
    """Extract text from PDF bytes page by page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        # Clean the text
        text = clean_text(text)

        # Skip empty or very short pages
        if len(text) < 100:
            continue

        pages.append({
            "title": title,
            "url": url,
            "page": i + 1,
            "text": text
        })

    print(f"  ✅ Extracted {len(pages)} pages from {title}")
    return pages

def load_all_books() -> list:
    """Download and extract all books from Azure."""
    print("☁️  Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    print("✅ Connected!\n")

    all_pages = []

    for book in BOOKS:
        print(f"📖 Processing: {book['title']}")

        # Download PDF from Azure
        pdf_bytes = download_blob(blob_service_client, book["blob_name"])

        # Extract text
        pages = extract_text_from_pdf(pdf_bytes, book["title"], book["url"])
        all_pages.extend(pages)
        print()

    return all_pages

def save_pages(pages: list, path: str):
    """Save extracted pages to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Load all books from Azure
    pages = load_all_books()

    # Summary
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"📊 Total pages extracted: {len(pages)}")
    print(f"📊 Total characters: {total_chars:,}")
    print()

    # Show breakdown per book
    from collections import Counter
    counts = Counter(p["title"] for p in pages)
    for title, count in counts.items():
        print(f"  📚 {title}: {count} pages")

    # Save to data folder
    output_path = "data/book_pages.json"
    save_pages(pages, output_path)
    print(f"\n💾 Saved to {output_path}")