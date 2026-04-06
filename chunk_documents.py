import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_articles(path: str) -> list:
    """Load articles from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_articles(articles: list) -> list:
    """Split articles into smaller chunks."""

    # This splitter tries to split on paragraphs first,
    # then sentences, then words — keeping chunks meaningful
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # max characters per chunk
        chunk_overlap=50,      # overlap between chunks
        separators=["\n\n", "\n", ". ", " "]
    )

    all_chunks = []

    for article in articles:
        chunks = splitter.split_text(article["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{article['title']}_{i}",   # unique ID
                "title": article["title"],           # source article
                "url": article["url"],               # source URL
                "chunk_index": i,                    # position in article
                "text": chunk                        # the actual text
            })

        print(f"✅ {article['title']} → {len(chunks)} chunks")

    return all_chunks

def save_chunks(chunks: list, path: str):
    """Save chunks to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print("📂 Loading articles...")
    articles = load_articles("data/articles.json")

    print(f"\n✂️  Chunking {len(articles)} articles...\n")
    chunks = chunk_articles(articles)

    output_path = "data/chunks.json"
    save_chunks(chunks, output_path)

    print(f"\n🎉 Done!")
    print(f"📊 Total chunks: {len(chunks)}")
    print(f"📊 Avg chunk size: {sum(len(c['text']) for c in chunks) // len(chunks)} chars")
    print(f"💾 Saved to {output_path}")