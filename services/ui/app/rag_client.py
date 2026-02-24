import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")


def rag_search(q: str, k: int = 8) -> dict:
    r = requests.get(f"{API_BASE_URL}/rag/search", params={"q": q, "k": k}, timeout=30)
    r.raise_for_status()
    return r.json()


def rag_answer(q: str, k: int = 8):
    r = requests.get(f"{API_BASE_URL}/rag/answer", params={"q": q, "k": k}, timeout=60)
    r.raise_for_status()
    return r.json()
