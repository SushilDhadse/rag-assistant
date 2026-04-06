import wikipediaapi
import json
import os

# Topics to fetch — Data Engineering & AI related
TOPICS = [
    "Data engineering",
    "Extract, transform, load",
    "Data pipeline",
    "Snowflake Inc",
    "Apache Airflow",
    "Retrieval-augmented generation",
    "Large language model",
    "Vector database",
    "Embedding (machine learning)",
    "Python (programming language)",
    "Data warehouse",
    "Data lakehouse",
    "Apache Spark",
    "Apache Kafka",
    "Distributed systems",
    "Feature store",
    "MLOps",
    "Prompt engineering",
    "Transformer architecture",
    "Semantic search",
    "Knowledge graph",
    "CI/CD pipeline",
    "Data governance",
    "Data mesh",
    "Vector similarity search",
    "Fine-tuning large language models",
    "Tokenization",
    "Graph database"
]

def fetch_article(topic: str, wiki: wikipediaapi.Wikipedia) -> dict | None:
    """Fetch a single Wikipedia article."""
    page = wiki.page(topic)

    if not page.exists():
        print(f"Article not found: {topic}")
        return None

    print(f"Fetched: {topic} ({len(page.text)} characters)")

    return {
        "title": page.title,
        "url": page.fullurl,
        "text": page.text
    }

def fetch_all_articles():
    """Fetch all articles and save to data folder."""

    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="rag-assistant/1.0"
    )

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    articles = []
    for topic in TOPICS:
        article = fetch_article(topic, wiki)
        if article:
            articles.append(article)

    # Save all articles to a single JSON file
    output_path = "data/articles.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Done! Saved {len(articles)} articles to {output_path}")
    print(f"📊 Total characters: {sum(len(a['text']) for a in articles):,}")

if __name__ == "__main__":
    fetch_all_articles()