from typing import Any
from openai import OpenAI
import chromadb

from app.configure import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)

from app.chunking import TextChunk

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

def get_embedding(texts: str) -> list[float]:
    """
    Convert text into an embedding vector using OpenAI embeddings.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return response.data[0].embedding


def get_chroma_collection():
    """
    Get or create a ChromaDB collection for document chunks.
    """
    return chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)


def document_filter(
    user_id: str, document_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a Chroma metadata filter.

    Every retrieval must be scoped by user_id so one user's documents cannot leaks
    into another user's query results.

    If document_id is provided, retrieval is further restricted to one document.

    """

    if document_id:
        return {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"document_id": {"$eq": document_id}},
            ]
        }
    return {"user_id": {"$eq": user_id}}


def delete_user_chunks(user_id: str) -> None:
    """
    Delete all chunks that belong to a specific user
    """
    collection = get_chroma_collection()
    collection.delete(
        where={"user_id": {"$eq": user_id}}
    )


def delete_document_chunks(
    user_id: str,
    document_id: str,
) -> None:
    """
    Delete existing chunks of a specific user's specific document.

    This is to make the ingestion idempotent: uploading the same PDF again replaces
    the old chunks instead of creating dulicates.
    """

    collection = get_chroma_collection()
    collection.delete(
        where=document_filter(user_id, document_id)
    )


def document_already_ingested(
    user_id: str,
    document_id: str,
) -> bool:
    """
    Check whether this user has already ingested this exact document.
    """
    collection = get_chroma_collection()

    result = collection.get(
        where=document_filter(user_id, document_id),
        limit=1,
    )

    return len(result.get("ids", [])) > 0


def add_chunks_to_vector_store(
    chunks: list[TextChunk], 
    user_id: str,
    document_id: str,
    document_name: str,
    document_hash: str
) -> int:
    """
    Embed page-aware, cross-page chunks and store them in ChromaDB.

    Each chunk is stored with user/document metadata so retrieval can be filtered
    by user_id and optionally document_id.
    """
    
    collection = get_chroma_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{document_id}-chunk-{chunk.chunk_index}"
        
        ids.append(chunk_id)
        documents.append(chunk.text)
        embedding = get_embedding(chunk.text)
        embeddings.append(embedding)

        metadatas.append(
            {
                "user_id": user_id,
                "document_id": document_id,
                "document_name": document_name,
                "document_hash": document_hash,
                "chunk_index": chunk.chunk_index,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "page_span": ",".join(str(page) for page in chunk.page_span),
            }
        )
    if not ids:
        return 0

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(chunks)


def search_similar_chunks(
    query: str, 
    top_k: int, 
    user_id: str,
    document_id: str | None = None,
    max_distance: float = 0.8
) -> list[dict[str, Any]]:
    """
    Search for chunks semantically similar to the query.

    Retrieval is always filtered by user_id. This is the core user-isolation rule.
    """

    collection = get_chroma_collection()

    query_embedding = get_embedding(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=document_filter(user_id, document_id),
    )

    matches = []

    if not result["ids"] or not result["ids"][0]:
        return matches

    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]
        
        if distance < max_distance:
            matches.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": distance,
                }
            )

    return matches




