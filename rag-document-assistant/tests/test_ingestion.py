from app.ingestion import extract_text_from_pdf
from app.chunking import chunk_text

text = extract_text_from_pdf('/Users/yuezf/Downloads/wsj.pdf')

# print(text[: 1000])
# print(len(text))

chunks = chunk_text(text, 20, 2)
print(f"Number of chunks: {len(chunks)}")
print(f"Fist chunk: {chunks[0]}")