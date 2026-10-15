"""
Rank fusion (pure functions — ทดสอบได้โดยไม่ต้องมี OpenSearch)

baseline: RRF แบบน้ำหนักเท่ากัน, ดึงมาเพียง k*2 = 10 ต่อ retriever แล้วตัดเหลือ 5
improved: Weighted RRF + candidate pool ใหญ่ขึ้น (30 ต่อ retriever) เพื่อให้ reranker
          มีตัวเลือกมากพอ (Anthropic ใช้ pool 150 → rerank → 20)
"""


def weighted_rrf(rankings, weights=None, k=60):
    """
    rankings: list ของผลค้นหาแต่ละวิธี [(doc_id, score, source), ...] เรียงจากดีไปแย่
    weights : น้ำหนักต่อวิธี (default เท่ากันหมด)
    RRF(d) = Σ_i  w_i / (k + rank_i(d))     โดย rank เริ่มที่ 1

    ใช้ "อันดับ" ไม่ใช้ "คะแนน" เพราะคะแนน BM25 (0-∞) กับ cosine (0-1) อยู่คนละสเกล
    เทียบตรง ๆ ไม่ได้ — อันดับเป็นสเกลเดียวกันเสมอ
    """
    weights = weights or [1.0] * len(rankings)
    fused, sources, trace = {}, {}, {}
    for r_idx, (results, w) in enumerate(zip(rankings, weights)):
        for rank, (doc_id, score, src) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
            sources.setdefault(doc_id, src)
            trace.setdefault(doc_id, {})[r_idx] = rank
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(d, s, sources[d], trace[d]) for d, s in ordered]


def merge_overlap(prev: str, nxt: str) -> str:
    """ต่อ chunk ที่อยู่ติดกัน โดยตัดส่วน overlap ที่ซ้ำออก (ซ้ำกันระดับบรรทัด)"""
    prev_lines = prev.split("\n")
    next_lines = nxt.split("\n")
    best = 0
    for n in range(1, min(len(prev_lines), len(next_lines)) + 1):
        if prev_lines[-n:] == next_lines[:n]:
            best = n
    return "\n".join(prev_lines + next_lines[best:])
