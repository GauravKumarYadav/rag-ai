from typing import Dict, List, Optional

from app.models.schemas import RetrievalHit
from app.rag.vector_store import VectorStore, get_vector_store


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def search(
        self, query: str, top_k: int = 4, metadata_filters: Optional[Dict] = None
    ) -> List[RetrievalHit]:
        return self.store.query(query=query, top_k=top_k, where=metadata_filters, collection="documents")


def get_retriever() -> Retriever:
    return Retriever(get_vector_store())

