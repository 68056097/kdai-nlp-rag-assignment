"""
Unit tests ที่รันได้โดยไม่ต้องมี OpenSearch / โมเดล / API key

    pytest -q
"""
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import evaluate, is_relevant, load_eval_set          # noqa: E402
from improved.chunking import chunk_markdown, load_and_chunk, parse_sections  # noqa: E402
from improved.config import SETTINGS as S                               # noqa: E402
from improved.fusion import merge_overlap, weighted_rrf                 # noqa: E402
from improved.thai_text import lucene_standard_like_tokenize, tokenize  # noqa: E402

PATHS = sorted(glob.glob(str(ROOT / "corpus_input" / "*.md")))


def _chunks():
    ch, _ = load_and_chunk(PATHS, max_chars=S.chunk_max_chars, overlap_chars=S.chunk_overlap_chars,
                           min_chars=S.chunk_min_chars, skip_sections=S.skip_sections)
    return ch


def test_thai_tokenizer_splits_words():
    toks = tokenize("วิธีการรักษาโรคหัดเยอรมันในหญิงตั้งครรภ์")
    assert "โรคหัดเยอรมัน" in toks and "ตั้งครรภ์" in toks
    assert "หัดเยอรมัน" in toks and "หัด" in toks          # decompound คำประสม
    assert "อหิ" not in tokenize("อหิวาตกโรค")              # ไม่แตกมั่ว
    # standard analyzer มัดทั้งประโยคเป็น token เดียว — ปัญหาที่เห็นในสไลด์หน้า 24
    assert lucene_standard_like_tokenize("และวิธีการรักษา") == ["และวิธีการรักษา"]


def test_chunk_sizes_bounded():
    ch = _chunks()
    assert len(ch) > 100
    assert max(len(c.text) for c in ch) <= S.chunk_max_chars


def test_breadcrumb_always_starts_with_document_title():
    for c in _chunks():
        assert c.headers[0] == c.title, c.breadcrumb


def test_boilerplate_sections_skipped():
    for c in _chunks():
        assert not any(s in h for s in ("เอกสารอ้างอิง", "เรื่องที่เกี่ยวข้อง") for h in c.headers)
        assert "https://" not in c.text


def test_every_gold_phrase_exists_in_some_chunk():
    """eval set ต้องตอบได้จาก chunk ของระบบ improved (ไม่งั้นการวัดผลไม่ยุติธรรม)"""
    ch = _chunks()
    for item in load_eval_set(ROOT / "eval" / "eval_set.json"):
        assert any(is_relevant(c.text, item["gold_phrases"]) for c in ch if c.doc_id == item["doc"]), item["id"]


def test_parse_sections_header_stack():
    md = "# A\nintro\n## B\nbody b\n### C\nbody c\n## D\nbody d\n"
    secs = parse_sections(md)
    assert [h for h, _ in secs] == [["A"], ["A", "B"], ["A", "B", "C"], ["A", "D"]]


def test_long_paragraph_split_on_word_boundary():
    text = "# T\n" + "การรักษาโรคหัดเยอรมัน" * 100
    ch = chunk_markdown(text, "x.md", max_chars=300, overlap_chars=50, min_chars=10)
    assert all(len(c.text) <= 300 for c in ch)
    assert all(c.text.startswith(("การ", "โรค")) for c in ch)   # ตัดที่ขอบคำ ไม่ตัดกลางคำ


def test_weighted_rrf():
    sparse = [("a", 10, {}), ("b", 5, {})]
    dense = [("b", .9, {}), ("c", .8, {})]
    fused = weighted_rrf([sparse, dense], k=60)
    assert fused[0][0] == "b"                      # อยู่ในทั้ง 2 list ได้คะแนนสะสม
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-9
    fused_w = weighted_rrf([sparse, dense], weights=[1.0, 0.0], k=60)
    assert fused_w[0][0] == "a"                    # ปิด dense (น้ำหนัก 0) → เหลือลำดับของ BM25


def test_merge_overlap():
    assert merge_overlap("x\ny\nz", "y\nz\nw") == "x\ny\nz\nw"
    assert merge_overlap("x", "w") == "x\nw"


def test_metrics():
    es = [{"id": "1", "gold_phrases": ["เป้า"]}, {"id": "2", "gold_phrases": ["ไม่มี"]}]
    m, per_q = evaluate({"1": ["ก", "มีเป้าหมาย"], "2": ["ข"]}, es)
    assert per_q == {"1": 2, "2": None}
    assert m["recall@1"] == 0 and m["recall@3"] == 0.5 and abs(m["mrr@10"] - 0.25) < 1e-9
