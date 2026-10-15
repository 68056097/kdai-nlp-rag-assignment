"""
ตัวชี้วัด retrieval (เหมือนที่ Anthropic ใช้ในสไลด์หน้า 18-19)

ความเกี่ยวข้อง (relevance) ของ chunk: ใน eval_set.json แต่ละคำถามมี gold_phrases
= ข้อความสั้น ๆ ที่ยกมาจากเอกสารจริง ซึ่ง "ต้องอยู่ใน chunk" ถึงจะตอบคำถามได้
chunk ใดมี gold phrase อย่างน้อย 1 อัน → ถือว่าเกี่ยวข้อง

    Recall@k  (hit rate)  : สัดส่วนคำถามที่มี chunk เกี่ยวข้องอย่างน้อย 1 ชิ้นใน top-k
    Failed@k              : 1 - Recall@k  (= "Failed Retrievals @k" ในกราฟของ Anthropic)
    MRR@k                 : เฉลี่ยของ 1/อันดับของ chunk เกี่ยวข้องชิ้นแรก (วัดว่า "ขึ้นมาเร็วแค่ไหน")
    ctx_chars@k           : จำนวนตัวอักษรรวมใน top-k (ต้นทุน context ที่ LLM ต้องอ่าน)
    recall@{B}ch          : ยุติธรรมกว่า Recall@k เมื่อเทียบ chunk คนละขนาด — เอาผลค้นหามาต่อกันตามอันดับ
                            แล้วตัดที่ B ตัวอักษร (เหมือนงบ context ของ LLM) ถามว่า gold phrase อยู่ในส่วนที่
                            LLM ได้อ่านหรือไม่ (chunk ใหญ่ได้ Recall@k สูงง่าย เพราะ 1 chunk = เนื้อหาหลายเรื่อง)
"""
import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def is_relevant(chunk_text: str, gold_phrases) -> bool:
    t = norm(chunk_text)
    return any(norm(p) in t for p in gold_phrases)


def load_eval_set(path="eval/eval_set.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


BUDGETS = (1500, 3000)


def found_within_budget(texts, gold_phrases, budget):
    joined = norm("\n\n".join(texts))[:budget]
    return any(norm(p) in joined for p in gold_phrases)


def evaluate(ranked_texts_per_q, eval_set, ks=(1, 3, 5, 10, 20)):
    """ranked_texts_per_q: {qid: [chunk_text อันดับ 1, 2, ...]}"""
    out = {f"recall@{k}": 0.0 for k in ks}
    out.update({f"ctx_chars@{k}": 0.0 for k in (5,)})
    out["mrr@10"] = 0.0
    per_q = {}
    n = len(eval_set)
    for item in eval_set:
        texts = ranked_texts_per_q.get(item["id"], [])
        flags = [is_relevant(t, item["gold_phrases"]) for t in texts]
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        per_q[item["id"]] = first
        for k in ks:
            out[f"recall@{k}"] += 1.0 if first and first <= k else 0.0
        out["mrr@10"] += (1.0 / first) if first and first <= 10 else 0.0
        out["ctx_chars@5"] += sum(len(t) for t in texts[:5])
        for b in BUDGETS:
            out[f"recall@{b}ch"] = out.get(f"recall@{b}ch", 0.0) + \
                (1.0 if found_within_budget(texts, item["gold_phrases"], b) else 0.0)
    for key in out:
        out[key] /= n
    for k in ks:
        out[f"failed@{k}"] = 1.0 - out[f"recall@{k}"]
    return out, per_q


def to_markdown(rows, ks=(1, 3, 5, 10, 20)):
    """rows: [(system_name, metrics_dict)]"""
    head = ("| ระบบ | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR@10 | Failed@5 | " +
            " | ".join(f"Recall@{b:,}ตัวอักษร" for b in BUDGETS) + " | ตัวอักษรใน top-5 |")
    sep = "|" + "---|" * (len(ks) + 4 + len(BUDGETS))
    lines = [head, sep]
    for name, m in rows:
        lines.append(f"| {name} | " + " | ".join(f"{m[f'recall@{k}']*100:.1f}%" for k in ks) +
                     f" | {m['mrr@10']:.3f} | {m['failed@5']*100:.1f}% | " +
                     " | ".join(f"{m[f'recall@{b}ch']*100:.1f}%" for b in BUDGETS) +
                     f" | {m['ctx_chars@5']:,.0f} |")
    return "\n".join(lines)
