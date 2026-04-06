import json

with open("data/articles.json", "r") as f:
    articles = json.load(f)

print(f"Total articles: {len(articles)}")
print(f"Total characters: {sum(len(a['text']) for a in articles):,}")
print("\nArticles fetched:")
for a in articles:
    print(f"  ✅ {a['title']} — {len(a['text']):,} chars")