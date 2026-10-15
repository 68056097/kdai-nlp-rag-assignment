"""
ค่าตั้งค่าทั้งหมดของ Improved RAG (ปรับได้ผ่าน environment variable)

หลักการ: ทุก "ปุ่มปรับ" ที่ต่างจาก baseline อยู่ในไฟล์นี้ไฟล์เดียว
เพื่อให้ทำ ablation (ปิด/เปิดทีละเทคนิค) แล้ววัดผลได้ง่าย
"""
import os
from dataclasses import dataclass, field


def _env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return cast(value)


@dataclass
class Settings:
    # ---------- Infrastructure ----------
    opensearch_host: str = field(default_factory=lambda: _env("OPENSEARCH_HOST", "localhost"))
    opensearch_port: int = field(default_factory=lambda: _env("OPENSEARCH_PORT", 9200, int))
    # ใช้ index เดียวเก็บทั้ง dense + sparse (baseline แยก 2 index ซึ่งอาจไม่ sync กัน)
    index_name: str = field(default_factory=lambda: _env("RAG_INDEX", "improved-rag-index"))

    # ---------- LLM (ใช้ตัวเดียวกับ baseline ทุกประการ — ไม่เปลี่ยน LLM) ----------
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen/qwen-2.5-72b-instruct"))
    answer_temperature: float = field(default_factory=lambda: _env("ANSWER_TEMPERATURE", 0.3, float))

    # ---------- (1) Chunking ----------
    chunk_max_chars: int = field(default_factory=lambda: _env("CHUNK_MAX_CHARS", 700, int))
    chunk_overlap_chars: int = field(default_factory=lambda: _env("CHUNK_OVERLAP_CHARS", 120, int))
    chunk_min_chars: int = field(default_factory=lambda: _env("CHUNK_MIN_CHARS", 80, int))
    # หัวข้อ boilerplate ที่ไม่มีเนื้อหาทางการแพทย์ (ลด noise ในการค้นหา)
    skip_sections: tuple = ("เอกสารอ้างอิง", "เรื่องที่เกี่ยวข้อง", "เมดไทย")

    # ---------- (2) Contextual chunk (Anthropic Contextual Retrieval) ----------
    use_context: bool = field(default_factory=lambda: _env("USE_CONTEXT", True, bool))
    # ส่ง "เอกสารเต็ม" ให้ LLM (baseline ส่งแค่ตัว chunk เอง = bug)
    context_doc_max_chars: int = field(default_factory=lambda: _env("CONTEXT_DOC_MAX_CHARS", 24000, int))
    context_cache_path: str = field(default_factory=lambda: _env("CONTEXT_CACHE", "cache/contexts.json"))

    # ---------- (3) Embedding ----------
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-m3"))
    embed_max_seq_len: int = 1024
    embed_batch_size: int = 16

    # ---------- (4) Search ----------
    candidates_per_retriever: int = field(default_factory=lambda: _env("CANDIDATES", 30, int))
    rrf_k: int = 60
    weight_dense: float = field(default_factory=lambda: _env("WEIGHT_DENSE", 1.0, float))
    weight_sparse: float = field(default_factory=lambda: _env("WEIGHT_SPARSE", 1.0, float))

    # ---------- (5) Reranking ----------
    use_rerank: bool = field(default_factory=lambda: _env("USE_RERANK", True, bool))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    rerank_top_n: int = field(default_factory=lambda: _env("RERANK_TOP_N", 5, int))
    # คะแนน (0-1) ต่ำกว่านี้ถือว่า "ไม่เกี่ยวข้อง" ตัดทิ้ง ไม่ส่งให้ LLM
    rerank_min_score: float = field(default_factory=lambda: _env("RERANK_MIN_SCORE", 0.05, float))

    # ---------- (6) Context assembly ----------
    expand_neighbors: bool = field(default_factory=lambda: _env("EXPAND_NEIGHBORS", True, bool))
    max_context_tokens: int = field(default_factory=lambda: _env("MAX_CONTEXT_TOKENS", 3500, int))


SETTINGS = Settings()
