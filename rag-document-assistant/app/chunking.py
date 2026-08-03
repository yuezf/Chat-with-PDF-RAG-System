from dataclasses import dataclass
from app.ingestion import PageText


@dataclass
class TextChunk:
    """
    A chunk of text that may come from one page or multiple page
    """
    text: str
    start_page: int
    end_page: int
    page_span: list[int]
    chunk_index: int


@dataclass
class PageSpan:
    page_number: int
    start_char: int
    end_char: int


def build_document_text_with_page_spans(
    pages_collection: list[PageText],
    page_separator: str = "\n\n",
    ) -> tuple[str, list[PageSpan]]:
    """
    1st Combine all the texts in the document into one big string. Return this
        string.
    
    2nd For each page in the original document, record its start and end positions
        (using the {PageSpan object) in this big string. Return a list of PageSpan
        objects
    """
    document_parts = []
    page_spans = []

    curr_offset = 0

    for page in pages_collection:
        if document_parts:
            document_parts.append(page_separator)
            curr_offset += len(page_separator)
        
        start_char = curr_offset
        document_parts.append(page.text)
        curr_offset += len(page.text)
        end_char = curr_offset

        page_spans.append(
            PageSpan(
                page_number=page.page_number,
                start_char=start_char,
                end_char=end_char,
            )
        )

    return "".join(document_parts), page_spans


def get_pages_for_chunk(
    chunk_start: int, chunk_end: int, page_spans: list[PageSpan],
    ) -> list[int]:
    """
    Return all PDF page numbers touched by the chunk's cahracter range
    """

    pages = []

    for span in page_spans:
        if chunk_start < span.end_char and chunk_end > span.start_char:
            pages.append(span.page_number)

    return pages


def chunking_across_pages(
    pages_collection: list[PageText],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[TextChunk]:
    """
    Chunk a PDF across page boundaries while preserving page metadata.

    The PDF is chunked in a way that useful information starts at the bottom of
    one page and continues at the top of the next page can be extracted in the
    same chunk.
    """

    if not pages_collection:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap > chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_index = 0

    document_text, page_spans = build_document_text_with_page_spans(pages_collection)
    
    start = 0
        
    while start < len(document_text):
        end = start + chunk_size
        chunk_text = document_text[start:end].strip()
        
        if chunk_text:
            pages = get_pages_for_chunk(
                chunk_start=start,
                chunk_end=end,
                page_spans=page_spans,
                )

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_page=pages[0],
                    end_page=pages[-1],
                    page_span=pages,
                )
            )
            chunk_index += 1
        
        if end == len(document_text):
            break
        
        start = end - overlap
    
    return chunks   