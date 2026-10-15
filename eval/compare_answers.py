"""
เปรียบเทียบ "คำตอบสุดท้าย" ของ baseline กับ improved แบบเคียงข้างกัน (ใช้ LLM ตัวเดียวกัน)

    python eval/compare_answers.py                 # ชุดคำถามของครู 5 ข้อ
    python eval/compare_answers.py --eval-set      # + คำถามใน eval_set.json (เรียก LLM มากขึ้น)
→ results/answers_side_by_side.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", action="store_true")
    args = ap.parse_args()

    from improved.ask import DEFAULT_QUESTIONS
    from improved.pipeline import ImprovedRAG
    from onlysearchAuthenticRAG import AuthenticSearchRAG

    questions = list(DEFAULT_QUESTIONS)
    if args.eval_set:
        questions += [q["question"] for q in json.loads(Path("eval/eval_set.json").read_text("utf-8"))]

    base, imp = AuthenticSearchRAG(), ImprovedRAG()
    md, dump = ["# คำตอบ Baseline vs Improved (LLM เดียวกัน: qwen/qwen-2.5-72b-instruct)", ""], []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        b = base.search_for_question(q, k=5)
        m = imp.answer(q, verbose=False)
        base_chars = sum(len(r["content"]) + len(r.get("context", "")) for r in b["results"])
        imp_chars = sum(len(s["text"]) for s in m["sources"])
        md += [f"## {i}. {q}", "",
               f"**Baseline** (context ≈ {base_chars:,} ตัวอักษร จาก {len(b['results'])} chunks)", "",
               b["answer"], "",
               f"**Improved** (context ≈ {imp_chars:,} ตัวอักษร จาก {len(m['sources'])} แหล่ง)", "",
               m["answer"], "", "แหล่งที่มา (improved): " +
               "; ".join(f"[{s['n']}] {s['breadcrumb']} (rerank={s['rerank_score']})" for s in m["sources"]),
               "", "---", ""]
        dump.append({"question": q, "baseline": b, "improved": m})
    Path("results").mkdir(exist_ok=True)
    Path("results/answers_side_by_side.md").write_text("\n".join(md), encoding="utf-8")
    Path("results/answers_side_by_side.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print("บันทึก results/answers_side_by_side.md")


if __name__ == "__main__":
    main()
