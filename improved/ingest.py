"""
นำเข้าเอกสาร (เทียบเท่า Hands-on 1 / authenticRAG.py)

    python -m improved.ingest                       # ทุกไฟล์ใน corpus_input/
    USE_CONTEXT=0 python -m improved.ingest         # ablation: ไม่สร้าง context (ไม่ต้องเรียก LLM)
"""
import argparse
import glob

from .config import SETTINGS
from .pipeline import ImprovedRAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_input/*.md")
    ap.add_argument("--no-recreate", action="store_true", help="ไม่ลบ index เดิมก่อนนำเข้า")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.corpus))
    if not paths:
        raise SystemExit(f"ไม่พบไฟล์ {args.corpus}")
    rag = ImprovedRAG(load_reranker=False, need_llm=SETTINGS.use_context)
    rag.ingest(paths, recreate=not args.no_recreate)
    print(f"[done] index '{SETTINGS.index_name}' มี {rag.store.count()} chunks")


if __name__ == "__main__":
    main()
