"""Knowledge base search over resolution summaries (Chroma + OpenAI embeddings).

Hybrid search: semantic similarity from Chroma, blended with keyword overlap
against each article's hand-written keyword list.
"""
import json
import re
from functools import lru_cache

import chromadb
from openai import OpenAI

from helpdesk.config import get_settings
from helpdesk.tools.schemas import KBHit, SearchKBInput, SearchKBOutput

COLLECTION = "kb_articles"
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


def _embed(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)
    resp = client.embeddings.create(model=s.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def _article_text(a: dict) -> str:
    return f"{a['title']}. Applies when: {a['applies_when']} Resolution: {a['resolution']}"


def build_index(articles: list[dict]) -> int:
    """(Re)create the Chroma collection from KB articles. Returns doc count."""
    s = get_settings()
    texts = [_article_text(a) for a in articles]
    embeddings = _embed(texts)  # before touching the index, so an API failure leaves the old one intact
    client = chromadb.PersistentClient(path=str(s.chroma_dir))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[a["article_id"] for a in articles],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"issue_type": a["issue_type"], "title": a["title"], "resolution": a["resolution"]}
            for a in articles
        ],
    )
    _collection.cache_clear()
    return col.count()


def index_count() -> int:
    """Documents in the KB index; 0 if it was never built."""
    try:
        return chromadb.PersistentClient(path=str(get_settings().chroma_dir)).get_collection(COLLECTION).count()
    except Exception:  # missing collection / empty dir
        return 0


@lru_cache
def _collection():
    return chromadb.PersistentClient(path=str(get_settings().chroma_dir)).get_collection(COLLECTION)


@lru_cache
def _keywords() -> dict[str, set[str]]:
    articles = json.loads(get_settings().kb_articles_path.read_text())
    return {a["article_id"]: set(a["keywords"]) for a in articles}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search(query: str, top_k: int = 3) -> list[dict]:
    """Return top_k articles ranked by blended semantic + keyword score."""
    embedding = _embed([query])
    try:
        col = _collection()
        res = col.query(query_embeddings=embedding, n_results=col.count())
    except Exception:  # index rebuilt by another process (prepare_data): reopen once
        _collection.cache_clear()
        col = _collection()
        res = col.query(query_embeddings=embedding, n_results=col.count())
    q_tokens = _tokens(query)
    hits = []
    for article_id, distance, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], strict=True):
        semantic = 1 - distance
        kw = _keywords().get(article_id, set())
        keyword = len(q_tokens & kw) / len(kw) if kw else 0.0
        hits.append({
            "article_id": article_id,
            "issue_type": meta["issue_type"],
            "title": meta["title"],
            "resolution": meta["resolution"],
            "score": round(SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * min(1.0, keyword * 3), 4),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def search_kb(inp: SearchKBInput) -> SearchKBOutput:
    """Registry handler for the search_kb tool."""
    return SearchKBOutput(hits=[KBHit(**h) for h in search(inp.query, inp.top_k)])
