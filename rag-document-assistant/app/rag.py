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


def format_page_range(metadata) -> str:
    """
    Turn the page range of a chunk into a string
    """
    start_page = metadata.get("start_page")
    end_page = metadata.get("end_page")

    if start_page is None or end_page is None:
        return "unknown page"

    if start_page == end_page:
        return f"page {start_page}"

    return f"pages {start_page}-{end_page}"


def generate_answer(
    query: str,
    top_k: int,
    user_id: str,
    document_id: str | None = None,
) -> dict:
    """
    Generate an answer using retrieved document chunks as context.

    Retrieval is scoped by user_id and optionally document_id.
    """ 

    retrieved_chunks = search_similar_chunks(query, top_k, user_id, document_id)
    context_blocks = []
    sources = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        page_range = format_page_range(metadata)

        context_blocks.append(
            f"""Source {i}
            Document: {metadata["document_name"]}
            Document ID: {metadata["document_id"]}
            Location: {page_range}
            Chunk index: {metadata["chunk_index"]}
            
            {chunk["text"]}"""
        )
        sources.append(
            {
                "source_id": i,
                "document_name": metadata["document_name"],
                "document_id": metadata["document_id"],
                "chunk_index": metadata["chunk_index"],
                "start_page": metadata.get("start_page"),
                "end_page": metadata.get("end_page"),
                "page_span": metadata.get("page_span"),
                "text_preview": chunk["text"][:100],
                "distance": chunk["distance"],
            }
        )
    
    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are a helpful document assistant.

    Answer the user's question using ONLY the context below.

    If the context does not contain enough information, say:
    "I don't have enough information in the uploaded documents to answer that."

    When possible, mention the source number you use to support your points.
    

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "user_id": user_id,
        "document_id": document_id,
        "answer": answer,
        "sources": sources,
    }