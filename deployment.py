from pipeline import rag_refresh_pipeline

if __name__ == "__main__":
    rag_refresh_pipeline.serve(
        name="rag-weekly-refresh",
        cron="0 6 * * 1",        # Every Monday at 6am
        description="Weekly RAG knowledge base refresh — fetches Wikipedia articles and PDFs from Azure, re-embeds everything into ChromaDB",
        tags=["rag", "chromadb", "weekly"]
    )