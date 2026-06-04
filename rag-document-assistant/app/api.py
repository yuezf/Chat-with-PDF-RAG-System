import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Annotated, Any
import chromadb

from app.configure import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

from app.ingestion import extract_text_from_pdf
from app.chunking import chunk_text
from app.vector_store import get_chroma_collection, add_chunks_to_vector_store, search_similar_chunks
from app.rag import generate_answer

app = FastAPI(title="RAG Document Assistant")


class ChunkingParameter(BaseModel):
    chunk_size: int
    overlap: int


class SearchRequest(BaseModel):
    queries: list[str]
    top_k: int


class AskRequest(BaseModel):
    query: str
    top_k: int


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_pdf(
    files: Annotated[list[UploadFile], File()], 
    chunk_size: int = Form(default=500), 
    overlap: int = Form(default=50)):

    collection = get_chroma_collection()

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only pdfs are supported")

        collection.delete(where={"document_name": file.filename})
    
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                shutil.copyfileobj(file.file,  temp)
                temp_path = temp.name
            
            texts = extract_text_from_pdf(temp_path)
            chunks = chunk_text(texts, chunk_size, overlap)

            add_chunks_to_vector_store(chunks, str(file.filename))

            Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    return {
        "filename": file.filename,
        "num_characters": len(texts),
        "num_chunks": len(chunks),
        "preview": chunks[0][:20],
    }


@app.get("/")
async def main():
    content = """
<body>
<h2>Ingest PDFs</h2>
<form action="/ingest" enctype="multipart/form-data" method="post">
    <label>Select PDF files:</label><br>
    <input name="files" type="file" multiple accept=".pdf"><br><br>
    <label>Chunk size:</label><br>
    <input name="chunk_size" type="number" value="500"><br><br>
    <label>Overlap:</label><br>
    <input name="overlap" type="number" value="50"><br><br>
    <input type="submit" value="Upload">
</form>
</body>
    """
    return HTMLResponse(content=content)


@app.post("/search")
async def retrieve_documents(request: SearchRequest) -> dict[str, dict[str, Any]]:
    try:
        result = {}
        for i, query in enumerate(request.queries):
            matches = search_similar_chunks(query, request.top_k)
            result[f"query-{i}"] = {
                "query": query,
                "top_k": request.top_k,
                "result": matches
            }
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
def get_answer_from_llm(request: AskRequest) -> dict[str, Any]:
    try:
        result = generate_answer(request.query, request.top_k)
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
        "items": all_items, 
    }


@app.delete("/collection")
def clear_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        return {"message": "Collection cleared successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




