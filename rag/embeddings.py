from openai import OpenAI
from django.conf import settings


client = OpenAI(api_key=settings.OPENAI_API_KEY)


def get_embedding(text: str) -> list:
    """Matn uchun embedding olish"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """Bir nechta matnlar uchun embedding olish"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
