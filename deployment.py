from pipeline import rag_pipeline

if __name__ == "__main__":
    rag_pipeline.serve(
        name="rag-daily-refresh",
        cron="0 6 * * 1-5",        # Everyday at 6am
        description="Daily RAG knowledge base refresh — fetches Wikipedia articles and PDFs from Azure, re-embeds everything into ChromaDB",
        tags=["rag", "Pinecone", "daily"]
    )