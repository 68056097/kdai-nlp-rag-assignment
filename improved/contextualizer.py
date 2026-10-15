"""
Contextual Chunk Generation (หัวใจของ Anthropic Contextual Retrieval)

ปัญหาของ baseline (authenticRAG.py บรรทัด ~199):
        full_doc_content = doc.page_content   # ถ้าไม่มีเอกสารเต็ม ใช้ chunk เอง
    → prompt ที่ควรเป็น <document>เอกสารเต็ม</document><chunk>...</chunk>
      กลายเป็น <document>chunk</document><chunk>chunk เดิม</chunk>
    LLM จึงไม่มีทางรู้ว่า chunk นี้อยู่ส่วนไหนของเอกสาร/เป็นของโรคอะไร
    context ที่ได้เป็นแค่ "สรุปซ้ำ" ของ chunk เอง ไม่ใช่ "การวางตำแหน่ง" (situate) ตามแนวคิดต้นฉบับ
    นอกจากนี้ prompt เป็นภาษาอังกฤษ → Qwen มักตอบ context เป็นภาษาอังกฤษ
    ซึ่ง BM25 จับคู่กับคำถามภาษาไทยไม่ได้

วิธีแก้:
    1) ส่ง "เอกสารเต็ม" จริงเข้าไป (ถ้ายาวเกินงบ ใช้สารบัญหัวข้อ + หน้าต่างรอบ chunk)
    2) สั่งให้ตอบเป็นภาษาไทย ระบุชื่อโรคและหัวข้อ + คำค้นที่คนน่าจะใช้ถาม
    3) วางเอกสารไว้ที่ต้น prompt คงที่ทุก chunk ของเอกสารเดียวกัน (เอื้อต่อ prompt caching ของ provider)
    4) cache ผลลง JSON → รัน ingest ซ้ำไม่ต้องเสียเงินเรียก LLM ใหม่
"""
import hashlib
import json
import re
from pathlib import Path

from tqdm import tqdm

PROMPT_VERSION = "v2-th"

DOC_PROMPT = """<document>
{document}
</document>"""

CHUNK_PROMPT = """Here is the chunk we want to situate within the whole document:
<chunk>
หัวข้อ: {breadcrumb}
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences, in Thai, same language as the document) \
to situate this chunk within the overall document for the purposes of improving search retrieval \
of the chunk. Mention the disease/topic name and which part of the document this chunk belongs to, \
and include key terms (Thai and English synonyms) a user might search for. \
Answer only with the succinct context and nothing else."""


def _doc_view(full_doc: str, chunk_text: str, max_chars: int) -> str:
    """เอกสารเต็มถ้าพอดีงบ; ไม่งั้นใช้ สารบัญหัวข้อ + ข้อความรอบ ๆ ตำแหน่ง chunk"""
    if len(full_doc) <= max_chars:
        return full_doc
    outline = "\n".join(l for l in full_doc.splitlines() if l.startswith("#"))
    anchor = full_doc.find(chunk_text[:60])
    anchor = max(anchor, 0)
    budget = max_chars - len(outline) - 50
    start = max(0, anchor - budget // 2)
    window = full_doc[start:start + budget]
    return f"[สารบัญเอกสาร]\n{outline}\n\n[ข้อความบางส่วนรอบ chunk]\n...{window}..."


class ContextGenerator:
    def __init__(self, llm, cache_path: str, doc_max_chars: int = 24000):
        self.llm = llm
        self.doc_max_chars = doc_max_chars
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, chunk):
        raw = f"{PROMPT_VERSION}|{chunk.doc_id}|{chunk.breadcrumb}|{chunk.text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def generate(self, chunk, full_doc: str) -> str:
        key = self._key(chunk)
        if key in self.cache:
            return self.cache[key]
        doc = _doc_view(full_doc, chunk.text, self.doc_max_chars)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            # ส่วน <document> อยู่หน้าสุดและเหมือนกันทุก chunk ของเอกสารเดียวกัน (prefix คงที่)
            {"role": "user", "content": DOC_PROMPT.format(document=doc) + "\n\n" +
                CHUNK_PROMPT.format(breadcrumb=chunk.breadcrumb, chunk=chunk.text)},
        ]
        ctx = self.llm.chat(messages, max_tokens=200, temperature=0.1)
        ctx = re.sub(r"\s+", " ", ctx).strip()
        self.cache[key] = ctx
        return ctx

    def contextualize_all(self, chunks, full_docs: dict):
        for i, ch in enumerate(tqdm(chunks, desc="Generating chunk contexts")):
            ch.context = self.generate(ch, full_docs[ch.doc_id])
            if i % 10 == 0:
                self._save()
        self._save()
        return chunks
