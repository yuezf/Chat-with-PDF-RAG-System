from typing import Any
from openai import OpenAI
import chromadb

from app.configure import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)

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


def add_chunks_to_vector_store(chunks: list[str], document_name: str) -> int:
    """
    Embed chunks and store them in ChromaDB.
    """
    
    collection = get_chroma_collection()

    ids = []
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        id = f"{document_name}-chunk-{i}"
        ids.append(id)

        embedding = get_embedding(chunk)
        embeddings.append(embedding)

        metadatas.append(
            {
                "document_name": document_name,
                "chunk_index": i,
                # "text": chunk
            }
        )
    print(f"Before add: {collection.count()}")

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"After add: {collection.count()}")

    return len(chunks)


def search_similar_chunks(query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Search for chunks semantically similar to the query.
    """
    collection = get_chroma_collection()

    query_embedding = get_embedding(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    matches = []

    for i in range(len(result["ids"][0])):
        if result["distances"][0][i] < 0.8:
            matches.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    # "embedding": result["embeddings"][0][i],
                    "metadatas": result["metadatas"][0][i],
                    "distance": result["distances"][0][i],
                }
            )

    return matches




