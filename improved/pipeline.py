"""
ImprovedRAG — ประกอบทุกชิ้นเข้าด้วยกัน

INGEST:  markdown → structure-aware chunks → LLM context (เอกสารเต็ม) → embed(หัวข้อ+context+chunk)
         → OpenSearch index เดียว (dense + Thai-tokenized BM25)
QUERY :  คำถาม → [BM25 ไทย top30] + [dense top30] → Weighted RRF → cross-encoder rerank
         → ตัดที่ไม่เกี่ยวข้อง (threshold) → top5 → ต่อ chunk ข้างเคียง → จัดงบ token
         → prompt พร้อมเลขอ้างอิง → Qwen (ตัวเดียวกับ baseline) → คำตอบ + แหล่งที่มา
"""
import json
from pathlib import Path

from .chunking import load_and_chunk
from .config import SETTINGS
from .fusion import weighted_rrf, merge_overlap
from .store import OpenSearchStore

ANSWER_SYSTEM = (
    "คุณเป็นผู้ช่วยตอบคำถามด้านสุขภาพจากเอกสารอ้างอิงที่ให้มาเท่านั้น "
    "ตอบเป็นภาษาไทย กระชับ ถูกต้อง และอ้างอิงหมายเลขแหล่งข้อมูล เช่น [1] [2] ท้ายประโยคที่ใช้ข้อมูลนั้น "
    "ห้ามแต่งข้อมูลที่ไม่มีในเอกสาร หากเอกสารไม่มีข้อมูลพอให้ตอบ ให้บอกตรง ๆ ว่าไม่พบข้อมูลในเอกสาร"
)

ANSWER_USER = """เอกสารอ้างอิง:
{sources}

คำถาม: {question}

คำตอบ (อ้างอิง [หมายเลข] ด้วย):"""


class ImprovedRAG:
    def __init__(self, settings=SETTINGS, load_reranker=None, need_llm=True):
        self.s = settings
        from .models import Embedder
        self.embedder = Embedder(settings.embed_model, settings.embed_max_seq_len, settings.embed_batch_size)
        self.store = OpenSearchStore(settings.opensearch_host, settings.opensearch_port, settings.index_name)
        self.reranker = None
        use_rerank = settings.use_rerank if load_reranker is None else load_reranker
        if use_rerank:
            from .models import Reranker
            self.reranker = Reranker(settings.rerank_model)
        self.llm = None
        if need_llm:
            from .llm import QwenClient
            self.llm = QwenClient(settings)

    # ================================================================ INGEST
    def ingest(self, md_paths, recreate=True, dump_chunks="cache/chunks.json"):
        s = self.s
        chunks, full_docs = load_and_chunk(
            md_paths, max_chars=s.chunk_max_chars, overlap_chars=s.chunk_overlap_chars,
            min_chars=s.chunk_min_chars, skip_sections=s.skip_sections)
        print(f"[chunking] {len(md_paths)} files → {len(chunks)} chunks")

        if s.use_context:
            from .contextualizer import ContextGenerator
            if self.llm is None:
                raise RuntimeError("USE_CONTEXT=1 ต้องมี OPENROUTER_API_KEY")
            gen = ContextGenerator(self.llm, s.context_cache_path, s.context_doc_max_chars)
            gen.contextualize_all(chunks, full_docs)

        passages = [c.passage(with_context=s.use_context) for c in chunks]
        embeddings = self.embedder.embed_documents(passages)
        if recreate:
            self.store.recreate_index(self.embedder.dim)
        ok, errors = self.store.bulk_index(chunks, embeddings)
        print(f"[index] {ok} docs indexed into '{s.index_name}', errors={len(errors) if errors else 0}")

        if dump_chunks:   # เก็บไว้ดู/นำเสนอว่า chunk + context หน้าตาเป็นอย่างไร
            Path(dump_chunks).parent.mkdir(parents=True, exist_ok=True)
            Path(dump_chunks).write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False,
                                                    indent=1), encoding="utf-8")
        return chunks

    # ================================================================ RETRIEVE
    @staticmethod
    def _passage_from_src(src, with_context=True):
        parts = [f"หัวข้อ: {src.get('breadcrumb', '')}"]
        if with_context and src.get("context"):
            parts.append(f"บริบท: {src['context']}")
        parts.append(src.get("content", ""))
        return "\n".join(parts)

    def retrieve(self, query: str, mode: str = "hybrid_rerank", top_n=None, apply_threshold=True):
        """
        mode: "sparse" | "dense" | "hybrid" | "hybrid_rerank"
        คืน list ของ dict ที่มี trace ทุกขั้น (อันดับ BM25 / dense / RRF / rerank) — ใช้อธิบายในห้องเรียนได้
        """
        s = self.s
        n_cand = s.candidates_per_retriever
        top_n = top_n or s.rerank_top_n

        if mode == "sparse":
            hits = self.store.sparse_search(query, n_cand, use_context=s.use_context)
            return [{"id": i, "score": sc, "src": src, "trace": {"bm25_rank": r}}
                    for r, (i, sc, src) in enumerate(hits, 1)][:top_n]
        q_vec = self.embedder.embed_query(query)
        if mode == "dense":
            hits = self.store.dense_search(q_vec, n_cand)
            return [{"id": i, "score": sc, "src": src, "trace": {"dense_rank": r}}
                    for r, (i, sc, src) in enumerate(hits, 1)][:top_n]

        sparse = self.store.sparse_search(query, n_cand, use_context=s.use_context)
        dense = self.store.dense_search(q_vec, n_cand)
        fused = weighted_rrf([sparse, dense], [s.weight_sparse, s.weight_dense], k=s.rrf_k)
        results = [{"id": d, "score": sc, "src": src,
                    "trace": {"bm25_rank": tr.get(0), "dense_rank": tr.get(1), "rrf_rank": r,
                              "rrf_score": round(sc, 5)}}
                   for r, (d, sc, src, tr) in enumerate(fused, 1)]

        if mode == "hybrid" or self.reranker is None:
            return results[:top_n]

        # ---- Reranking: ให้ cross-encoder ให้คะแนนความเกี่ยวข้อง (query, passage) ใหม่ทั้ง pool
        passages = [self._passage_from_src(r["src"], s.use_context) for r in results]
        scores = self.reranker.score(query, passages)
        for r, sc in zip(results, scores):
            r["score"] = sc
            r["trace"]["rerank_score"] = round(sc, 4)
        results.sort(key=lambda r: r["score"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["trace"]["rerank_rank"] = rank
        if apply_threshold:
            kept = [r for r in results if r["score"] >= s.rerank_min_score]
            results = kept or results[:1]          # กันกรณีตัดหมด
        return results[:top_n]

    # ================================================================ CONTEXT
    def _count_tokens(self, text):
        if self.reranker is not None:
            return self.reranker.count_tokens(text)
        return int(len(text) / 2.5) + 1           # ประมาณการภาษาไทย (baseline ใช้ split() ซึ่งผิดกับไทย)

    def build_context(self, hits):
        """
        - ต่อ chunk ข้างเคียง (หัวข้อเดียวกัน) ให้ข้อมูลไม่ขาดตอน
        - รวม chunk ที่ติดกันเป็นก้อนเดียว ลบส่วน overlap ซ้ำ
        - คุมงบ token จริง (นับด้วย tokenizer) — baseline นับด้วย .split() ซึ่งภาษาไทยไม่มีช่องว่าง
          ทำให้ "ประมาณ token" ผิดไปมาก
        """
        s = self.s
        blocks = []          # [{doc_id, section_id, positions:set, texts:{pos:text}, src, best_rank}]
        for rank, h in enumerate(hits):
            src = h["src"]
            pos = src["position"]
            wanted = {pos}
            if s.expand_neighbors and rank < 3:      # ขยายเฉพาะผลอันดับต้น ๆ
                wanted |= {pos - 1, pos + 1}
            neighbors = self.store.get_by_positions(src["doc_id"], wanted) if len(wanted) > 1 else {pos: src}
            neighbors = {p: v for p, v in neighbors.items() if v["section_id"] == src["section_id"]}
            neighbors[pos] = src
            # รวมกับ block เดิมถ้าอยู่ section เดียวกันและตำแหน่งต่อกัน
            target = None
            for b in blocks:
                if b["doc_id"] == src["doc_id"] and b["section_id"] == src["section_id"] and \
                        (min(neighbors) - 1 <= max(b["texts"]) and max(neighbors) + 1 >= min(b["texts"])):
                    target = b
                    break
            if target is None:
                target = {"doc_id": src["doc_id"], "section_id": src["section_id"], "texts": {},
                          "src": src, "rank": rank, "rerank_score": h["trace"].get("rerank_score")}
                blocks.append(target)
            for p, v in neighbors.items():
                target["texts"][p] = v["content"]

        sources, used = [], 0
        for i, b in enumerate(sorted(blocks, key=lambda b: b["rank"]), 1):
            text = ""
            for p in sorted(b["texts"]):
                text = merge_overlap(text, b["texts"][p]) if text else b["texts"][p]
            entry = f"[{i}] หัวข้อ: {b['src']['breadcrumb']}\n{text}"
            t = self._count_tokens(entry)
            if used + t > s.max_context_tokens and sources:
                break
            sources.append({"n": i, "breadcrumb": b["src"]["breadcrumb"], "doc_id": b["doc_id"],
                            "text": text, "rerank_score": b["rerank_score"], "entry": entry})
            used += t
        return sources, used

    # ================================================================ ANSWER
    def answer(self, question: str, verbose=True):
        hits = self.retrieve(question)
        sources, n_tok = self.build_context(hits)
        if verbose:
            print(f"\n[retrieve] {len(hits)} chunks → {len(sources)} source blocks (~{n_tok} tokens)")
            for h in hits:
                t = h["trace"]
                print(f"  bm25#{t.get('bm25_rank')} dense#{t.get('dense_rank')} rrf#{t.get('rrf_rank')} "
                      f"→ rerank={t.get('rerank_score')} | {h['src']['breadcrumb'][:80]}")
        prompt = ANSWER_USER.format(sources="\n\n".join(s["entry"] for s in sources), question=question)
        answer = self.llm.chat([{"role": "system", "content": ANSWER_SYSTEM},
                                {"role": "user", "content": prompt}],
                               max_tokens=1000, temperature=self.s.answer_temperature)
        return {
            "question": question,
            "answer": answer,
            "context_tokens": n_tok,
            "sources": [{k: v for k, v in s.items() if k != "entry"} for s in sources],
            "retrieval_trace": [{"chunk_id": h["id"], "breadcrumb": h["src"]["breadcrumb"],
                                 "context": h["src"].get("context", ""), **h["trace"]} for h in hits],
        }
