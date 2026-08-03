from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file page by page
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    reader = PdfReader(path)
    pages_collection = []
    
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        
        if text:
            pages_collection.append(PageText(page_number=index, text=text))

    return pages_collection
