from docx import Document
import re


def load_docx(file_path: str) -> str:
    """Word fayldan matnni o'qish"""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Matnni bo'laklarga bo'lish

    Args:
        text: Bo'laklarga bo'linadigan matn
        chunk_size: Har bir bo'lak uzunligi (so'zlarda) - optimal: 500
        overlap: Bo'laklar orasidagi overlap (so'zlarda) - optimal: 100

    Returns:
        Bo'laklar ro'yxati

    Optimallashtirilgan parametrlar:
    - 500 so'zlik chunks: yaxshiroq semantic coherence
    - 100 so'zlik overlap: kontekst davomiyligini saqlash
    """
    # Matnni paragraphlarga bo'lish
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        words = para.split()
        para_length = len(words)

        # Agar paragraf juda katta bo'lsa, uni ham bo'lish
        if para_length > chunk_size:
            # Avvalgi chunk ni saqlash
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

            # Katta paragrafni bo'lish
            for i in range(0, para_length, chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunks.append(' '.join(chunk_words))
        else:
            # Chunk ga sig'adimi tekshirish
            if current_length + para_length > chunk_size:
                # Chunk ni saqlash
                chunks.append(' '.join(current_chunk))

                # Overlap qo'shish
                overlap_words = ' '.join(current_chunk).split()[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words + [para]
                current_length = len(overlap_words) + para_length
            else:
                current_chunk.append(para)
                current_length += para_length

    # Oxirgi chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_rules_file(file_path: str) -> list:
    """
    Rules faylni o'qish va bo'laklarga bo'lish

    Returns:
        [{'id': 0, 'text': '...', 'metadata': {...}}, ...]
    """
    text = load_docx(file_path)
    chunks = chunk_text(text)

    processed_chunks = []
    for i, chunk in enumerate(chunks):
        processed_chunks.append({
            'id': f"chunk_{i}",
            'text': chunk,
            'metadata': {
                'source': file_path,
                'chunk_index': i
            }
        })

    return processed_chunks
