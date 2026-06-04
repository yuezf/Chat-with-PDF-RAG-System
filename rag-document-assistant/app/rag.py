from openai import OpenAI
from app.vector_store import search_similar_chunks

from app.configure import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


def generate_answer(query: str, top_k: int) -> dict:
    """
    Generate an answer using retrieved document chunks as context.
    """ 
    retrieved_chunks = search_similar_chunks(query, top_k)
    context_blocks = []
    sources = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"Source {i}:\n{chunk["text"]}"
        )
        sources.append(
            {
                "source_id": i,
                "document_name": chunk["metadatas"]["document_name"],
                "chunk_index": chunk["metadatas"]["chunk_index"],
                "text_preview": chunk["text"][:250],
                "distance": chunk["distance"],
            }
        )
    
    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are a helpful document assistant.

    Answer the user's question using ONLY the context below.

    If the context does not contain enough information, say:
    "I don't have enough information in the uploaded documents to answer that."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    reponse = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    answer = reponse.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "sources": sources
    }