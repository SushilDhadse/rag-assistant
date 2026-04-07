import json
from chunk_documents import chunk_articles
from embed_and_store import embed_and_store
from sentence_transformers import SentenceTransformer

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":

    # Step 1: Load Wikipedia articles
    print("📂 Loading Wikipedia articles...")
    wiki_articles = load_json("data/articles.json")
    print(f"✅ {len(wiki_articles)} Wikipedia articles loaded")

    # Step 2: Load book pages
    print("\n📂 Loading book pages...")
    book_pages = load_json("data/book_pages.json")
    print(f"✅ {len(book_pages)} book pages loaded")

    # Step 3: Chunk everything together
    print("\n✂️  Chunking all documents...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    all_chunks = []

    # Chunk Wikipedia articles
    for article in wiki_articles:
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

    wiki_chunk_count = len(all_chunks)
    print(f"✅ Wikipedia: {wiki_chunk_count} chunks")

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

    book_chunk_count = len(all_chunks) - wiki_chunk_count
    print(f"✅ Books: {book_chunk_count} chunks")
    print(f"✅ Total: {len(all_chunks)} chunks")

    # Save combined chunks
    save_json(all_chunks, "data/all_chunks.json")
    print(f"\n💾 Saved to data/all_chunks.json")

    # Step 4: Re-embed everything
    print("\n🔢 Embedding all chunks into ChromaDB...")
    collection = embed_and_store(all_chunks)

    print(f"\n🎉 Knowledge base built!")
    print(f"📊 Total chunks in ChromaDB: {collection.count()}")
    print(f"   - Wikipedia: {wiki_chunk_count} chunks")
    print(f"   - Books: {book_chunk_count} chunks")