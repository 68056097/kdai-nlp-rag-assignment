"""
Offline study: ผลของ (ก) วิธี chunking และ (ข) วิธีตัดคำ ต่อ BM25 — รันได้ทันทีโดยไม่ต้องมี
OpenSearch / GPU / API key (ใช้สูตร BM25 แบบเดียวกับ Lucene/OpenSearch: k1=1.2, b=0.75)

    python eval/offline_bm25_study.py

แยกผลของ 2 ปัจจัยด้วยการทดลองแบบ 2x2:
                         standard tokenizer (baseline)   PyThaiNLP newmm (improved)
    baseline chunks             A                                B
    improved chunks             C                                D
    + E = D + แตกคำประสม (decompound) ซึ่งเป็นค่าที่ระบบ improved ใช้จริง
หมายเหตุ: ไม่รวม contextualized_content (ต้องใช้ LLM) และไม่รวม dense search (ต้องใช้ bge-m3)
ส่วนนั้นวัดด้วย eval/evaluate_retrieval.py ที่รันกับระบบจริง
"""
import glob
import math
from functools import partial
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import evaluate, load_eval_set, to_markdown           # noqa: E402
from improved.chunking import load_and_chunk                            # noqa: E402
from improved.config import SETTINGS as S                               # noqa: E402
from improved.thai_text import tokenize, lucene_standard_like_tokenize  # noqa: E402


class LuceneBM25:
    def __init__(self, docs_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for d in docs_tokens for t in set(d))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, q_tokens, k=20):
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q_tokens:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [i for s, i in scores[:k] if s > 0]


def baseline_chunks(paths):
    """จำลอง baseline: MarkdownNodeParser(chunk_size=256) จาก llama-index"""
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownNodeParser
    parser = MarkdownNodeParser(chunk_size=256)
    docs = [Document(text=Path(p).read_text(encoding="utf-8"), metadata={"source": p}) for p in paths]
    return [n.text for n in parser.get_nodes_from_documents(docs)]


def run():
    paths = sorted(glob.glob(str(ROOT / "corpus_input" / "*.md")))
    eval_set = load_eval_set(ROOT / "eval" / "eval_set.json")

    base = baseline_chunks(paths)
    imp_chunks, _ = load_and_chunk(paths, max_chars=S.chunk_max_chars, overlap_chars=S.chunk_overlap_chars,
                                   min_chars=S.chunk_min_chars, skip_sections=S.skip_sections)
    imp_index_text = [f"{c.breadcrumb}\n{c.text}" for c in imp_chunks]   # breadcrumb ถูก index ด้วย
    imp_display = [c.text for c in imp_chunks]

    def describe(name, texts):
        L = [len(t) for t in texts]
        return f"| {name} | {len(texts)} | {min(L):,} | {max(L):,} | {sum(L)//len(L):,} |"

    print("## ขนาด chunk\n\n| วิธี | จำนวน chunk | สั้นสุด | ยาวสุด | เฉลี่ย (ตัวอักษร) |\n|---|---|---|---|---|")
    print(describe("baseline MarkdownNodeParser(256)", base))
    print(describe("improved structure-aware (700/120)", imp_display))

    q_example = eval_set[0]["question"]
    print(f"\n## ตัวอย่างการตัดคำ query: \"{q_example}\"")
    print(f"- standard : {lucene_standard_like_tokenize(q_example)}")
    print(f"- PyThaiNLP: {tokenize(q_example, use_decompound=False)}")
    print(f"- PyThaiNLP + decompound: {tokenize(q_example)}")

    rows = []
    newmm = partial(tokenize, use_decompound=False)
    configs = [
        ("A: baseline chunks + standard tokenizer", base, base, lucene_standard_like_tokenize),
        ("B: baseline chunks + PyThaiNLP", base, base, newmm),
        ("C: improved chunks + standard tokenizer", imp_index_text, imp_display, lucene_standard_like_tokenize),
        ("D: improved chunks + PyThaiNLP", imp_index_text, imp_display, newmm),
        ("E: improved chunks + PyThaiNLP + decompound", imp_index_text, imp_display, tokenize),
    ]
    per_q_all = {}
    for name, index_texts, display, tok in configs:
        bm25 = LuceneBM25([tok(t) for t in index_texts])
        ranked = {q["id"]: [display[i] for i in bm25.search(tok(q["question"]), k=20)] for q in eval_set}
        m, per_q = evaluate(ranked, eval_set)
        rows.append((name, m))
        per_q_all[name[:1]] = per_q
    print("\n## BM25-only retrieval บน eval set (24 คำถาม)\n")
    print(to_markdown(rows))

    print("\n## อันดับของ chunk ที่ถูกต้อง รายคำถาม (— = ไม่อยู่ใน top-20)\n")
    print("| คำถาม | " + " | ".join(per_q_all) + " |\n|---|" + "---|" * len(per_q_all))
    for q in eval_set:
        print(f"| {q['id']} {q['question'][:40]} | " +
              " | ".join(str(per_q_all[k][q['id']] or "—") for k in per_q_all) + " |")


if __name__ == "__main__":
    run()
