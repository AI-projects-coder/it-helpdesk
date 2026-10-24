"""Settings loaded from environment / .env."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    judge_model: str
    embedding_model: str
    openai_base_url: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    kb_top_k: int = 3

    raw_csv: Path = DATA_DIR / "raw" / "golden_it_support_tickets.csv"
    golden_path: Path = DATA_DIR / "golden_subset.json"
    accounts_path: Path = DATA_DIR / "accounts.json"
    kb_articles_path: Path = DATA_DIR / "kb_articles.json"
    chroma_dir: Path = DATA_DIR / "chroma"
    sessions_dir: Path = DATA_DIR / "sessions"
    traces_dir: Path = DATA_DIR / "traces"
    eval_dir: Path = DATA_DIR / "eval"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        kb_top_k=int(os.getenv("KB_TOP_K", "3")),
    )


def api_key_problem(settings: Settings | None = None) -> str | None:
    """Human-readable reason the configured provider can't be called, or None if a key is set."""
    s = settings or get_settings()
    key = s.anthropic_api_key if s.llm_provider == "anthropic" else s.openai_api_key
    name = "ANTHROPIC_API_KEY" if s.llm_provider == "anthropic" else "OPENAI_API_KEY"
    if not key or key.endswith("..."):
        return f"{name} in .env is missing or still the placeholder. Set a real key, save .env, and restart."
    return None
