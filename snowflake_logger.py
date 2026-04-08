import os
import uuid
import json
import time
from dotenv import load_dotenv
from datetime import datetime, timezone
from snowflake.connector import connect
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Load variables from the .env file into os.environ
load_dotenv()

SNOWFLAKE_TABLE = "RAG_ASSISTANT.PIPELINES.PIPELINE_RUN_LOG"
CHAT_TABLE = "RAG_ASSISTANT.PIPELINES.RAG_CHAT_HISTORY"

def get_snowflake_conn():
    """Fetch Snowflake connection string from Azure Key Vault."""
    kv_url     = os.getenv("AZURE_KEY_VAULT_URL")
    credential = DefaultAzureCredential()
    client     = SecretClient(vault_url=kv_url, credential=credential)
    conn_str   = client.get_secret("SNOWFLAKE-CONNECTION-STRING").value

    params = {
        pair.split("=", 1)[0].strip().lower(): pair.split("=", 1)[1].strip()
        for pair in conn_str.split(";")
        if "=" in pair
    }
    
    return connect(**params)


class PipelineRunLogger:
    """Logs a single pipeline run to Snowflake."""

    def __init__(self):
        self.run_id      = str(uuid.uuid4())
        self.started_at  = datetime.now(timezone.utc)
        self.stats       = {
            "wiki_articles_fetched": 0,
            "pdfs_ingested":         0,
            "total_chunks_created":  0,
            "vectors_upserted":      0,
        }
        self.error_message = None

    def update(self, **kwargs):
        """Call this during the flow to accumulate stats."""
        self.stats.update(kwargs)

    def commit(self, status: str = "SUCCESS"):
        """Write the completed run record to Snowflake."""
        finished_at = datetime.now(timezone.utc)
        duration    = (finished_at - self.started_at).total_seconds()

        conn = get_snowflake_conn()
        try:
            conn.cursor().execute(f"""
                INSERT INTO {SNOWFLAKE_TABLE} (
                    run_id, run_started_at, run_finished_at, status,
                    wiki_articles_fetched, pdfs_ingested,
                    total_chunks_created, vectors_upserted,
                    pinecone_index_name, embedding_model,
                    error_message, duration_seconds
                ) VALUES (
                    %(run_id)s, %(started_at)s, %(finished_at)s, %(status)s,
                    %(wiki)s, %(pdfs)s, %(chunks)s, %(vectors)s,
                    %(index)s, %(model)s, %(error)s, %(duration)s
                )
            """, {
                "run_id":     self.run_id,
                "started_at": self.started_at,
                "finished_at":finished_at,
                "status":     status,
                "wiki":       self.stats["wiki_articles_fetched"],
                "pdfs":       self.stats["pdfs_ingested"],
                "chunks":     self.stats["total_chunks_created"],
                "vectors":    self.stats["vectors_upserted"],
                "index":      os.getenv("PINECONE_INDEX_NAME"),
                "model":      "all-MiniLM-L6-v2",
                "error":      self.error_message,
                "duration":   duration,
            })
            conn.commit()
        finally:
            conn.close()


def log_chat_turn(
    session_id:      str,
    question:        str,
    answer:          str,
    chunks:          list,
    response_time_ms: int,
):
    """Write a single Q&A turn + RAG sources to Snowflake."""
    sources_payload = [
        {
            "title": c["title"],
            "url":   c["url"],
            "score": round(c.get("score", 0), 4),
        }
        for c in chunks
    ]

    conn = get_snowflake_conn()
    try:
        conn.cursor().execute(f"""
            INSERT INTO {CHAT_TABLE} (
                message_id, session_id, asked_at, question, answer, 
                sources_used, num_chunks_used, top_source_title, response_time_ms
            )
            SELECT 
                %(m_id)s, %(s_id)s, %(asked)s, %(q)s, %(a)s, 
                PARSE_JSON(%(src)s), %(num)s, %(top)s, %(r_time)s
            """, {
                "m_id": str(uuid.uuid4()),
                "s_id": session_id,
                "asked": datetime.now(timezone.utc),
                "q": question,
                "a": answer,
                "src": json.dumps(sources_payload),
                "num": len(chunks),
                "top": chunks[0]["title"] if chunks else None,
                "r_time": response_time_ms
            })
        conn.commit()
    finally:
        conn.close()