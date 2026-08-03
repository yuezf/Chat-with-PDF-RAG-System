import hashlib
import uuid
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Annotated, Any
import chromadb

from app.configure import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

from app.ingestion import extract_pages_from_pdf
from app.chunking import chunking_across_pages
from app.vector_store import (
    get_chroma_collection,
    add_chunks_to_vector_store,
    search_similar_chunks,
    delete_user_chunks,
    delete_document_chunks,
    document_already_ingested,
)
from app.rag import generate_answer

app = FastAPI(title="RAG Document Assistant")


class ChunkingParameter(BaseModel):
    chunk_size: int
    overlap: int


class SearchRequest(BaseModel):
    queries: list[str]
    top_k: int
    user_id: str
    document_id: str | None = None


class AskRequest(BaseModel):
    query: str
    top_k: int
    user_id: str
    document_id: str | None = None


def compute_file_hash(file_path: str) -> str:
    """
    Generate a 64-character SHA256 hash string for the content of each document.

    If two documents share the exact same content, this hashed string would be
    the same. This identifies the actual document content, not just the file name.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def build_document_id(user_id: str, document_hash: str) -> str:
    """
    Generate a deterministic user_id that is unique for each user_id and document
    hash pair.
    
    Same user + same document bytes => same document_id
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{document_hash}")) 


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_pdf(
    files: Annotated[list[UploadFile], File()], 
    user_id: Annotated[str, Form()],
    chunk_size: Annotated[int, Form()] = 500, 
    overlap: Annotated[int, Form()] = 50,
    force_reprocess: Annotated[bool, Form()] = False,
):
    if not user_id.strip():
        raise HTTPException(code_status=400, detail="user_id is required")
    
    result = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only pdfs are supported")

        # collection.delete(where={"document_name": file.filename})
        temp_path = None
    
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                shutil.copyfileobj(file.file, temp)
                temp_path = temp.name

            document_hash = compute_file_hash(temp_path)
            document_id = build_document_id(user_id, document_hash)

            already_ingested = document_already_ingested(user_id, document_id)

            if already_ingested and not force_reprocess:
                result.append(
                    {
                        "filename": file.filename,
                        "user_id": user_id,
                        "document_id": document_id,
                        "document_hash": document_hash,
                        "status": "skipped_already_ingested",
                    }
                )
                continue

            if already_ingested and force_reprocess:
                delete_document_chunks(user_id, document_id)
                
            pages_collection = extract_pages_from_pdf(temp_path)
            chunks = chunking_across_pages(
            pages_collection, chunk_size, overlap,
            )

            num_chunks = add_chunks_to_vector_store(
                chunks=chunks, 
                user_id=user_id,
                document_id=document_id,
                document_name=file.filename,
                document_hash=document_hash,
            )

            result.append(
                {
                    "filename": file.filename,
                    "user_id": user_id,
                    "document_id": document_id,
                    "document_hash": document_hash,
                    "status": "force_reprocess" if already_ingested else "newly_ingested",
                    "num_pages_with_text": len(pages_collection),
                    "num_chunks": num_chunks,
                    "preview": chunks[0].text[:100] if chunks else "",
                    "first_chunk_page_range": {
                        "start_page": chunks[0].start_page,
                        "end_page": chunks[0].end_page,
                        "page_span": chunks[0].page_span,
                    } if chunks else None,
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
        
    return {"documents": result}


@app.get("/")
async def main():
    content = """
    <html>
        <body>
            <h2>Ingest PDFs</h2>
            <form action="/ingest" enctype="multipart/form-data" method="post">
                <label>User ID:</label>
                <input name="user_id" type="text" value="demo-user"><br><br>

                <label>Select PDF files:</label>
                <input name="files" type="file" multiple><br><br>

                <label>Chunk size:</label>
                <input name="chunk_size" type="number" value="500"><br><br>

                <label>Overlap:</label>
                <input name="overlap" type="number" value="50"><br><br>

                <label>Force reprocessing document (Can be used if you want to
                    re-chunk the existing documents)?</label>
                <input name="force_reprocess" type="checkbox" checked><br><br>

                <input type="submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)


@app.post("/search")
async def retrieve_documents(request: SearchRequest) -> dict[str, dict[str, Any]]:
    try:
        result = {}
        for i, query in enumerate(request.queries):
            matches = search_similar_chunks(
                query,
                request.top_k,
                request.user_id,
                request.document_id,
            )
            result[f"query-{i}"] = {
                "query": query,
                "top_k": request.top_k,
                "user_id": request.user_id,
                "document_id": request.document_id,
                "result": matches,
            }
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
def get_answer_from_llm(request: AskRequest) -> dict[str, Any]:
    try:
        result = generate_answer(
            request.query,
            request.top_k,
            request.user_id,
            request.document_id,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection-info")
def collection_info():
    collection = get_chroma_collection()
    all_items = collection.get()
    ids = all_items["ids"]
    return {
        # "total_chunks": len(ids),
        # "unique_chunks": len(set(ids)),
        # "has_duplicates": len(ids) != len(set(ids)),
        "count": collection.count(),
        "items": all_items, 
    }


@app.delete("/collection/user/{user_id}")
def clear_user_collection(user_id: str):
    try:
        if not user_id.strip():
            raise HTTPException(status_code=500, detail="missing user_id")

        delete_user_chunks(user_id)

        return {
            "message": f"All documents for {user_id} have been successfully deleted.",
            "user_id": user_id,
        }
    
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection/user/{user_id}/document/{document_id}")
def clear_user_collection(user_id: str, document_id: str):
    try:
        if not user_id.strip():
            raise HTTPException(status_code=500, detail="missing user_id")
        
        if not document_id.strip():
            raise HTTPException(status_code=500, detail="missing document_id")

        delete_document_chunks(user_id, document_id)

        return {
            "message": f"All documents for {user_id} have been successfully deleted.",
            "user_id": user_id,
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection")
def clear_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        return {"message": "Collection cleared successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




