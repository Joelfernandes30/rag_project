import os
import re
import glob
import time
from typing import List, Dict, Any
import pymupdf
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import RAG engine helpers
from rag_engine import (
    get_supabase_client,
    get_google_genai_client,
    get_google_embeddings,
    TABLE_NAME,
    EMBEDDING_DIMENSION
)

CHAPTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chapters")

# Target ONLY all 6 PDF files in chapters/ directory
ALL_6_PDF_FILES = [
    os.path.join(CHAPTERS_DIR, "Drug_Development_Book.pdf"),
    os.path.join(CHAPTERS_DIR, "Evidence_based_Ayurvedic_Practice.pdf"),
    os.path.join(CHAPTERS_DIR, "AYURVEDA_The_Science_of_LifeDossier.pdf"),
    os.path.join(CHAPTERS_DIR, "Medico-Ethno-Botanical-Survey-Programmme.pdf"),
    os.path.join(CHAPTERS_DIR, "CCRAS-Vision-Document-2030.pdf"),
    os.path.join(CHAPTERS_DIR, "05112021_Nutritional_Advocacy_in_Ayurveda.pdf")
]


def clean_pdf_text(text: str) -> str:
    """
    Cleans text extracted from PDFs:
    - Normalizes replacement characters (\ufffd, \u00a0).
    - Strips OCR header noise, standalone page numbers, and brackets.
    - Fixes broken hyphenations and normalizes whitespace.
    """
    if not text:
        return ""

    text = text.replace('\ufffd', ' ').replace('\u00a0', ' ')
    cleaned_lines = []
    
    for line in text.splitlines():
        trimmed = line.strip()
        if re.match(r'^={5,}$', trimmed):
            continue
        if re.match(r'^(CARAKA[-_]SAMHITĀ|CHIKITSĀSTHĀNAM|CHIKITSASTHANAM|\[\s*CH\.?|XXX\]|\d{3,4})$', trimmed, re.IGNORECASE):
            continue
        if re.match(r'^\d{1,4}$', trimmed):
            continue

        line = re.sub(r'\[\s*\d+[-–]?\d*\s*\]', '', line)

        if line.strip():
            cleaned_lines.append(line.strip())

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def ocr_page_with_gemini(page: pymupdf.Page) -> str:
    """
    Uses Google Gemini 2.5 Flash Vision API to perform OCR on scanned PDF pages.
    """
    try:
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        client = get_google_genai_client()
        
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Extract all readable text from this document page image cleanly and accurately.",
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            ]
        )
        return clean_pdf_text(res.text or "")
    except Exception as e:
        print(f"Gemini OCR notice: {e}")
        return ""


def parse_pdf_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts page-level text from PDF, applying Gemini Vision OCR for scanned pages.
    """
    filename = os.path.basename(file_path)
    clean_title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ")
    
    doc = pymupdf.open(file_path)
    total_pages = len(doc)
    print(f"\n[Parsing PDF] '{filename}' ({total_pages} pages)...", flush=True)
    
    segments = []
    ocr_count = 0
    
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        raw_text = page.get_text() or ""
        cleaned = clean_pdf_text(raw_text)
        
        # Fast direct text check; apply Gemini Vision OCR only if page has 0/minimal text
        if len(cleaned) < 30:
            cleaned = ocr_page_with_gemini(page)
            if cleaned:
                ocr_count += 1
                
        if cleaned and len(cleaned) > 30:
            segments.append({
                "text": cleaned,
                "section": "CCRAS Research Publication & Medical Reference",
                "chapter": clean_title,
                "overall_pages": f"1-{total_pages}",
                "page_number": str(page_num),
                "source_file": f"chapters/{filename}"
            })

    if ocr_count > 0:
        print(f"  -> Applied Gemini Vision OCR on {ocr_count} scanned pages in '{filename}'.", flush=True)

    return segments


def build_vector_database(clear_table: bool = True) -> Dict[str, Any]:
    """
    Ingests ONLY all 6 PDF books into Supabase pgvector with IMMEDIATE PER-FILE STREAMING.
    """
    sb = get_supabase_client()

    print("Targeting ALL 6 PDF books with Instant File Streaming:\n", flush=True)
    for f in ALL_6_PDF_FILES:
        print(f" - {os.path.basename(f)}", flush=True)

    if clear_table:
        try:
            print(f"\nClearing previous vector records from Supabase table '{TABLE_NAME}'...", flush=True)
            sb.table(TABLE_NAME).delete().neq("id", "none").execute()
            print("Table cleared. Beginning immediate real-time vector streaming...\n", flush=True)
        except Exception as clear_err:
            print(f"Notice while clearing table: {clear_err}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    total_indexed_all_files = 0
    chunk_counter = 0

    for file_path in ALL_6_PDF_FILES:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: Target PDF not found at {file_path}")
            continue

        filename = os.path.basename(file_path)
        segments = parse_pdf_file(file_path)
        
        file_records = []
        for seg in segments:
            chunks = text_splitter.split_text(seg["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{seg['chapter']}_p{seg['page_number']}_c{idx}_{chunk_counter}".replace(" ", "_")
                file_records.append({
                    "id": chunk_id,
                    "content": chunk,
                    "section": seg["section"],
                    "chapter": seg["chapter"],
                    "pages": str(seg["overall_pages"]),
                    "page_number": str(seg["page_number"]),
                    "source_file": seg["source_file"],
                    "chapter_label": f"{seg['section']} - {seg['chapter']}"
                })
                chunk_counter += 1

        print(f"Extracted {len(file_records)} chunks from '{filename}'. Generating 768-dim Google embeddings & streaming to Supabase...", flush=True)

        # STREAM THIS PDF FILE'S RECORDS IMMEDIATELY TO SUPABASE
        batch_size = 25
        for i in range(0, len(file_records), batch_size):
            batch_records = file_records[i : i + batch_size]
            batch_texts = [r["content"] for r in batch_records]
            
            embeddings = get_google_embeddings(batch_texts, dimension=EMBEDDING_DIMENSION)
            for idx_r, emb in enumerate(embeddings):
                batch_records[idx_r]["embedding"] = emb

            try:
                sb.table(TABLE_NAME).upsert(batch_records).execute()
                total_indexed_all_files += len(batch_records)
                print(f"[Supabase Live Stream] +{len(batch_records)} records uploaded from '{filename}' (Total in Database: {total_indexed_all_files})", flush=True)
            except Exception as upsert_err:
                print(f"Batch upsert error: {upsert_err}")
                raise upsert_err

    print(f"\nSUCCESS! Fully indexed total {total_indexed_all_files} PDF document chunks into Supabase.", flush=True)
    return {"status": "success", "document_count": total_indexed_all_files}


if __name__ == "__main__":
    build_vector_database(clear_table=True)
