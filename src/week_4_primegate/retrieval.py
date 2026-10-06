import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "entre_ch1"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def search_chapter(query: str, limit: int=5):
    results = client.query_points(
        collection_name=COLLECTION,
        query=models.Document(text=query, model=MODEL),
        with_payload=["text"],
        limit=limit,
    )
    return [hit.payload.get("text", "") for hit in results.points]