"""
การตัดคำภาษาไทยสำหรับ BM25 (lexical search)

ปัญหาของ baseline:
    OpenSearch `standard` analyzer ตัดคำตามช่องว่าง/เครื่องหมายวรรคตอน (Unicode UAX#29)
    ภาษาไทยไม่เว้นวรรคระหว่างคำ → ข้อความไทยทั้งก้อนกลายเป็น "term เดียว"
    (ดูสไลด์หน้า 24: term "และวิธีการรักษา" คือ 3 คำที่ถูกมัดรวมกัน)
    ผลคือ query "วิธีรักษา" แทบไม่ match อะไรเลย → BM25 ใช้กับภาษาไทยไม่ได้จริง

วิธีแก้:
    ตัดคำด้วย PyThaiNLP (engine newmm, dictionary-based maximal matching) ก่อนส่งเข้า index
    แล้วเก็บเป็นข้อความที่คั่นด้วยช่องว่าง ให้ OpenSearch ใช้ `whitespace` analyzer
    ทั้งตอน index และตอน query ใช้ฟังก์ชันเดียวกัน → token ตรงกันแน่นอน

    + Decompounding: dictionary ของ newmm มีคำประสมยาว ๆ เช่น "โรคหัดเยอรมัน" เป็นคำเดียว
      ถ้าผู้ใช้ถาม "หัดเยอรมัน" จะไม่ match → เราแตกคำประสมเพิ่ม (เก็บทั้งคำเต็มและคำย่อย)
      "โรคหัดเยอรมัน" → โรคหัดเยอรมัน, โรค, หัดเยอรมัน, หัด, เยอรมัน
      (แตกเฉพาะเมื่อทุกคำย่อยเป็นคำในพจนานุกรม กันการแตกมั่ว เช่น อหิวาตกโรค → อหิ/วา/ตก)
"""
import re
import unicodedata
from functools import lru_cache

from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie
from pythainlp.util import normalize as thai_normalize

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+")


@lru_cache(maxsize=1)
def _stopwords():
    # stopword ไทย + คำเชื่อมที่พบบ่อยในคำถาม
    extra = {"อย่างไร", "ไหม", "บ้าง", "อะไร", "ครับ", "คะ", "ค่ะ", "หรือไม่", "ยังไง", "แบบไหน"}
    return frozenset(thai_stopwords()) | extra


@lru_cache(maxsize=1)
def _trie():
    return Trie(thai_words())


@lru_cache(maxsize=1)
def _wordset():
    return frozenset(thai_words())


@lru_cache(maxsize=50000)
def decompound(token: str, depth: int = 2) -> tuple:
    """แตกคำประสมไทยเป็นคำย่อยที่อยู่ในพจนานุกรม (ไม่รวมตัวมันเอง)"""
    if depth == 0 or len(token) < 4 or not re.search(r"[\u0E00-\u0E7F]", token):
        return ()
    trie = _trie()
    if token not in _wordset():
        return ()
    trie.remove(token)
    try:
        parts = word_tokenize(token, custom_dict=trie, engine="newmm")
    finally:
        trie.add(token)
    if len(parts) < 2 or not all(p in _wordset() and len(p) >= 3 for p in parts):
        return ()
    out = []
    for p in parts:
        out.append(p)
        out.extend(decompound(p, depth - 1))
    return tuple(out)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    return thai_normalize(text)


def tokenize(text: str, remove_stopwords: bool = True, use_decompound: bool = True) -> list:
    """ตัดคำ → lowercase → ทิ้งเครื่องหมาย/ช่องว่าง/stopword → (แตกคำประสม)"""
    tokens = []
    stop = _stopwords()
    for tok in word_tokenize(clean_text(text), engine="newmm", keep_whitespace=False):
        tok = tok.strip().lower()
        if not tok or _PUNCT_RE.match(tok):
            continue
        if remove_stopwords and tok in stop:
            continue
        tokens.append(tok)
        if use_decompound:
            tokens.extend(t for t in decompound(tok) if not (remove_stopwords and t in stop))
    return tokens


def to_index_text(text: str) -> str:
    """ข้อความที่ตัดคำแล้ว คั่นด้วยช่องว่าง สำหรับ field *_tok ใน OpenSearch"""
    return " ".join(tokenize(text))


def lucene_standard_like_tokenize(text: str) -> list:
    """
    จำลองพฤติกรรม `standard` analyzer ของ baseline กับภาษาไทย (ใช้ใน offline study เท่านั้น)
    ตัดตามช่องว่างและเครื่องหมาย; ตัวอักษรไทยที่ติดกันทั้งก้อน = 1 token
    """
    return [t.lower() for t in re.findall(r"[0-9A-Za-z฀-๿]+", clean_text(text))]
