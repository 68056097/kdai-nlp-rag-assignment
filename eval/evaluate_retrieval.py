"""
วัดผล retrieval ของระบบจริง (baseline vs improved) บน OpenSearch

ต้องรัน ingest ของทั้ง 2 ระบบก่อน:
    python baseline/authenticRAG.py      # → anthropic-vector-index, anthropic-bm25-index
    python -m improved.ingest            # → improved-rag-index
แล้ว:
    python eval/evaluate_retrieval.py    # → results/retrieval_eval.md + .json

ไม่เรียก LLM ตอนประเมิน (วัดเฉพาะการค้นคืน) จึงไม่เสียค่า API
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))
os.chdir(ROOT)

from eval.metrics import evaluate, load_eval_set, to_markdown   # noqa: E402

K = 20


def eval_baseline(eval_set):
    os.environ.setdefault("OPENROUTER_API_KEY", "not-needed-for-retrieval")
    from onlysearchAuthenticRAG import AuthenticSearchRAG
    rag = AuthenticSearchRAG()
    systems = {
        "Baseline · BM25 (standard analyzer)": lambda q: rag.sparse_search(q, k=K),
        "Baseline · Dense": lambda q: rag.dense_search(q, k=K),
        # hybrid_search ของ baseline ดึง k*2 ต่อวิธี แล้วคืน k อันดับแรก — เรียกด้วย k=K เพื่อดูถึง @20
        "Baseline · Hybrid RRF (สไลด์หน้า 25)": lambda q: rag.hybrid_search(q, k=K),
    }
    out = []
    for name, fn in systems.items():
        ranked = {item["id"]: [src.get("content", "") for _, _, src in fn(item["question"])]
                  for item in eval_set}
        out.append((name, ranked))
    return out


def eval_improved(eval_set):
    from improved.pipeline import ImprovedRAG
    rag = ImprovedRAG(need_llm=False)
    modes = [
        ("Improved · BM25 ไทย (PyThaiNLP)", "sparse"),
        ("Improved · Dense", "dense"),
        ("Improved · Hybrid Weighted-RRF", "hybrid"),
        ("Improved · Hybrid + Rerank (ระบบเต็ม)", "hybrid_rerank"),
    ]
    out = []
    for name, mode in modes:
        if mode == "hybrid_rerank" and rag.reranker is None:
            continue
        ranked = {}
        for item in eval_set:
            hits = rag.retrieve(item["question"], mode=mode, top_n=K, apply_threshold=False)
            ranked[item["id"]] = [h["src"]["content"] for h in hits]
        out.append((name, ranked))
    return out


def main():
    eval_set = load_eval_set(ROOT / "eval" / "eval_set.json")
    runs = []
    for loader in (eval_baseline, eval_improved):
        try:
            runs.extend(loader(eval_set))
        except Exception as e:      # เช่น ยังไม่ได้ ingest ระบบใดระบบหนึ่ง
            print(f"[skip] {loader.__name__}: {e}")

    rows, per_q = [], {}
    for name, ranked in runs:
        m, pq = evaluate(ranked, eval_set)
        rows.append((name, m))
        per_q[name] = pq
    table = to_markdown(rows)
    print(table)

    Path("results").mkdir(exist_ok=True)
    md = ["# Retrieval evaluation (24 คำถาม, eval/eval_set.json)", "", table, "",
          "## อันดับของ chunk ที่ถูกต้อง รายคำถาม (— = ไม่อยู่ใน top-20)", "",
          "| คำถาม | " + " | ".join(f"S{i+1}" for i in range(len(rows))) + " |",
          "|---|" + "---|" * len(rows)]
    for item in eval_set:
        md.append(f"| {item['id']} {item['question']} | " +
                  " | ".join(str(per_q[n][item['id']] or "—") for n, _ in rows) + " |")
    md += ["", "S1..Sn = " + ", ".join(f"S{i+1}: {n}" for i, (n, _) in enumerate(rows))]
    Path("results/retrieval_eval.md").write_text("\n".join(md), encoding="utf-8")
    Path("results/retrieval_eval.json").write_text(
        json.dumps({n: m for n, m in rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nบันทึก results/retrieval_eval.md และ results/retrieval_eval.json")


if __name__ == "__main__":
    main()
