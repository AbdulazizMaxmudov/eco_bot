import chromadb
from chromadb.config import Settings
import os
from django.conf import settings as django_settings
from .embeddings import get_embedding, get_embeddings_batch
from .chunker import process_rules_file


# ChromaDB client
CHROMA_PATH = os.path.join(django_settings.BASE_DIR, 'chroma_db')

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Collection yaratish yoki olish
collection = client.get_or_create_collection(
    name="eco_rules",
    metadata={"description": "Ekologik qoidalar"}
)


def index_rules(file_path: str):
    """
    Rules faylni indekslash (vector DB ga yuklash)
    """
    # Avvalgi ma'lumotlarni o'chirish
    try:
        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])
    except Exception:
        pass

    # Chunks olish
    chunks = process_rules_file(file_path)

    if not chunks:
        print("Chunks topilmadi!")
        return

    # Embeddings olish
    texts = [chunk['text'] for chunk in chunks]
    embeddings = get_embeddings_batch(texts)

    # ChromaDB ga qo'shish
    collection.add(
        ids=[chunk['id'] for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[chunk['metadata'] for chunk in chunks]
    )

    print(f"{len(chunks)} ta chunk indekslandi!")


def search(query: str, n_results: int = 3) -> list:
    """
    Savol bo'yicha eng yaqin chunklar ni qidirish

    Args:
        query: Foydalanuvchi savoli
        n_results: Qaytariladigan natijalar soni

    Returns:
        [{text, score, metadata}, ...]
    """
    # Query embedding
    query_embedding = get_embedding(query)

    # Qidirish
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Natijalarni formatlash
    formatted_results = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                'text': doc,
                'score': results['distances'][0][i] if results['distances'] else None,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })

    return formatted_results


def get_context(query: str, n_results: int = 3) -> str:
    """
    Savol uchun kontekst olish (GPT ga yuborish uchun)
    """
    results = search(query, n_results)

    if not results:
        return ""

    context = "\n\n---\n\n".join([r['text'] for r in results])
    return context
