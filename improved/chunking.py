"""
Structure-aware chunking สำหรับ Markdown ภาษาไทย

ปัญหาของ baseline:
    `MarkdownNodeParser(chunk_size=256)` ตัดตามหัวข้อ markdown เท่านั้น — พารามิเตอร์ chunk_size
    ไม่มีผลจริง ทำให้ได้ chunk ขนาด 112 ถึง 8,404 ตัวอักษร (เฉลี่ย ~1,700) จากเอกสาร 4 ไฟล์ = 53 chunks
    - chunk ใหญ่มาก → embedding 1 เวกเตอร์ต้องแทนหลายเรื่อง ความหมาย "เจือจาง" (semantic dilution)
    - ส่งให้ LLM แล้วกินพื้นที่ context มาก แต่ส่วนที่ตอบคำถามจริงมีแค่ไม่กี่บรรทัด
    - หัวข้อ (header path) ไม่ได้ติดไปกับ chunk ย่อยที่อยู่ใต้หัวข้อ

วิธีแก้ (3 ชั้น):
    1) แบ่งตามหัวข้อก่อน และจำ "เส้นทางหัวข้อ" (breadcrumb) เช่น
       "หัดเยอรมัน › ภาวะแทรกซ้อนของโรคหัดเยอรมัน › ภาวะแทรกซ้อนที่พบได้ในหญิงตั้งครรภ์"
    2) ภายในหัวข้อ แบ่งย่อยตามย่อหน้า/bullet แล้ว "รวม" ให้ได้ขนาดไม่เกิน chunk_max_chars
       ถ้าย่อหน้าเดียวยาวเกิน ตัดที่ขอบ "คำ" ภาษาไทย (PyThaiNLP) ไม่ตัดกลางคำ
    3) มี overlap ระหว่าง chunk ติดกัน (~120 ตัวอักษร) กันข้อมูลขาดตอนที่รอยต่อ
    + ตัด boilerplate (เอกสารอ้างอิง/ลิงก์/โฆษณาเว็บ) ที่ไม่ช่วยตอบคำถาม
"""
import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pythainlp.tokenize import word_tokenize

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_URL_IN_PARENS_RE = re.compile(r"\(\s*https?://[^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_CITATION_RE = re.compile(r"\[\d+\]")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str          # ชื่อไฟล์ (stem) เช่น "1"
    source: str          # path ไฟล์
    title: str           # หัวข้อ H1 แรกของเอกสาร (ชื่อโรค)
    headers: list        # เส้นทางหัวข้อ
    section_id: int      # ลำดับหัวข้อในเอกสาร
    position: int        # ลำดับ chunk ในเอกสาร (ใช้หา chunk ข้างเคียง)
    text: str            # เนื้อหา chunk (ไม่รวม breadcrumb)
    context: str = ""    # context ที่ LLM สร้าง (Contextual Retrieval)
    extra: dict = field(default_factory=dict)

    @property
    def breadcrumb(self) -> str:
        return " › ".join(h for h in self.headers if h)

    def passage(self, with_context: bool = True) -> str:
        """ข้อความที่ใช้ทำ embedding และ rerank: หัวข้อ + context + เนื้อหา"""
        parts = [f"หัวข้อ: {self.breadcrumb}"]
        if with_context and self.context:
            parts.append(f"บริบท: {self.context}")
        parts.append(self.text)
        return "\n".join(parts)

    def to_dict(self):
        d = asdict(self)
        d["breadcrumb"] = self.breadcrumb
        return d


def _clean_line(line: str) -> str:
    if line.strip().startswith("โดย เมดไทย"):   # byline ผู้เขียน/วันที่ ไม่มีสาระ
        return ""
    line = _URL_IN_PARENS_RE.sub("", line)
    line = _URL_RE.sub("", line)
    line = _CITATION_RE.sub("", line)          # เลขอ้างอิง [2]
    line = re.sub(r"[ \t\u00a0]+", " ", line)
    return line.strip()


def parse_sections(markdown: str):
    """คืน list ของ (headers_path, body_text) ตามลำดับในเอกสาร"""
    sections = []
    stack = []            # [(level, title)]
    body = []

    def flush():
        text = "\n".join(body).strip()
        sections.append(([t for _, t in stack], text))
        body.clear()

    for raw in markdown.splitlines():
        m = _HEADER_RE.match(raw)
        if m:
            flush()
            level, title = len(m.group(1)), _clean_line(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        if raw.strip() in ("---", "***") or _TABLE_SEP_RE.match(raw) and "-" in raw:
            body.append("")           # ตัวคั่น = ขึ้นย่อหน้าใหม่
            continue
        body.append(_clean_line(raw))
    flush()
    return [(h, t) for h, t in sections if t]


def _paragraphs(text: str):
    """แยกเป็นหน่วยย่อย: ย่อหน้า (บรรทัดว่าง) และ bullet แต่ละข้อ"""
    units = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.splitlines() if l.strip()]
        buf = []
        for l in lines:
            if re.match(r"^\s*([-*•○]|\d+[.)])\s+", l) and buf:
                units.append(" ".join(buf))
                buf = []
            buf.append(l.strip())
        if buf:
            units.append(" ".join(buf))
    return [u for u in units if u.strip()]


def _split_long(unit: str, max_chars: int):
    """ย่อหน้าที่ยาวเกิน → ตัดที่ขอบช่องว่างก่อน ถ้ายังยาว ตัดที่ขอบคำไทย"""
    if len(unit) <= max_chars:
        return [unit]
    pieces, cur = [], ""
    # ภาษาไทยมักเว้นวรรคระหว่าง "ประโยค/วลี" → ใช้ช่องว่างเป็นขอบตัดอันดับแรก
    for seg in re.split(r"(?<=\s)", unit):
        if len(seg) > max_chars:               # วลียาวมากไม่มีช่องว่าง → ตัดตามคำ
            for w in word_tokenize(seg, engine="newmm"):
                if len(cur) + len(w) > max_chars and cur:
                    pieces.append(cur.strip()); cur = ""
                cur += w
            continue
        if len(cur) + len(seg) > max_chars and cur:
            pieces.append(cur.strip()); cur = ""
        cur += seg
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def _pack(units, max_chars: int, overlap_chars: int):
    """รวมหน่วยย่อยให้ได้ chunk ขนาดไม่เกิน max_chars พร้อม overlap ท้ายต่อหัว"""
    chunks, cur = [], []
    cur_len = 0
    for u in units:
        if cur and cur_len + len(u) + 1 > max_chars:
            chunks.append("\n".join(cur))
            # overlap: ยกหน่วยท้ายของ chunk ก่อนหน้ามาเป็นหัวของ chunk ถัดไป (ไม่เกิน overlap_chars)
            tail, tail_len = [], 0
            for prev in reversed(cur):
                if tail_len + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev); tail_len += len(prev) + 1
            cur, cur_len = tail, tail_len
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _short_title(title: str) -> str:
    """
    ชื่อบทความ medthai ยาวแบบ clickbait: "หัดเยอรมัน อาการ สาเหตุ และการรักษาโรคหัดเยอรมัน 10 วิธี !!"
    ถ้าติดชื่อยาวนี้ไปกับทุก chunk คำว่า อาการ/สาเหตุ/รักษา จะไปโผล่ในทุก chunk ของทุกโรค
    → เหลือแค่ชื่อโรค "หัดเยอรมัน" / "ต้อกระจก (Cataract)"
    """
    t = re.split(r"\s+อาการ", title, maxsplit=1)[0]
    t = re.sub(r"[!\s]+$", "", t).strip(" ,")
    return t if len(t) >= 3 else title


def chunk_markdown(markdown: str, source: str, max_chars=700, overlap_chars=120,
                   min_chars=80, skip_sections=()):
    doc_id = Path(source).stem
    sections = parse_sections(markdown)
    raw_title = sections[0][0][0] if sections and sections[0][0] else doc_id
    title = _short_title(raw_title)
    out = []
    position = 0
    pending = None  # หัวข้อที่สั้นมาก (เช่นบทนำ 1 บรรทัด) → รวมกับหัวข้อถัดไป

    for section_id, (headers, body) in enumerate(sections):
        if any(skip in h for skip in skip_sections for h in headers[1:] or headers):
            continue
        if len(body) < min_chars and pending is None and section_id + 1 < len(sections):
            pending = body
            continue
        if pending:
            body = pending + "\n\n" + body
            pending = None
        # เอกสารบางไฟล์ใช้ H1 ซ้ำกลางเรื่อง (เช่น "# วิธีป้องกันโรคหัดเยอรมัน") ทำให้ชื่อโรคหลุดจาก
        # เส้นทางหัวข้อ → เติมชื่อเอกสารไว้หน้าสุดเสมอ เพื่อให้ chunk รู้ว่าเป็นของโรคอะไร
        headers = list(headers)
        if headers and headers[0] == raw_title:
            headers[0] = title
        elif not headers or headers[0] != title:
            headers = [title] + headers
        units = []
        for p in _paragraphs(body):
            units.extend(_split_long(p, max_chars - overlap_chars))
        for text in _pack(units, max_chars, overlap_chars):
            if len(text.strip()) < 15:
                continue
            cid = hashlib.sha1(f"{doc_id}:{position}:{text}".encode("utf-8")).hexdigest()[:16]
            out.append(Chunk(chunk_id=cid, doc_id=doc_id, source=source, title=title,
                             headers=headers or [title], section_id=section_id,
                             position=position, text=text))
            position += 1
    return out


def load_and_chunk(paths, **kwargs):
    chunks, full_docs = [], {}
    for p in paths:
        md = Path(p).read_text(encoding="utf-8")
        full_docs[Path(p).stem] = md
        chunks.extend(chunk_markdown(md, str(p), **kwargs))
    return chunks, full_docs
