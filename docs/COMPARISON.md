# (3) ความแตกต่างระหว่าง Baseline (1) และ Improved (2)

เอกสารนี้ตอบ 3 คำถามของโจทย์ข้อ (3) ทีละองค์ประกอบ: **ต่างกันตรงไหน → ดีขึ้นอย่างไร → เพราะเหตุใด**
พร้อมชี้บรรทัดในโค้ด baseline และหลักฐานจากการทดลอง

หลักคิดทั้งหมดมาจากสไลด์หน้า 15: *"The success of a RAG system begins with the quality of its retriever"*
— เราไม่เปลี่ยน LLM (ยังเป็น `qwen/qwen-2.5-72b-instruct`) แต่ทำให้ LLM "ได้อ่านสิ่งที่ถูกต้อง ในปริมาณที่พอดี"

และแก้ข้อจำกัดของ Generic RAG ทั้ง 4 ข้อในสไลด์หน้า 14:

| ข้อจำกัด (สไลด์หน้า 14) | แก้ด้วย |
|---|---|
| สูญเสีย context / chunk ขาดความเชื่อมโยง | §1 chunking + breadcrumb + overlap, §2 context จากเอกสารเต็ม, §7 ต่อ chunk ข้างเคียง |
| เน้น dense search เกินไป | §4 ทำให้ BM25 ภาษาไทยใช้งานได้จริง (ใน baseline BM25 แทบไม่ทำงาน → ระบบเป็น dense-only โดยปริยาย) |
| การ scoring ผลลัพธ์จาก vector DB | §5 Weighted RRF + pool ใหญ่ขึ้น |
| ขาดการวัด relevance ระหว่างผลค้นกับคำถาม | §6 cross-encoder reranker + threshold |

---

## §0 หลักฐานจากผลลัพธ์ของ baseline เอง

ไฟล์ `authentic_rag_search_results.json` ใน repo ของครู (คัดลอกไว้ที่
[`results/baseline_reference_output_from_instructor_repo.json`](../results/baseline_reference_output_from_instructor_repo.json))
คือผลรัน `onlysearchAuthenticRAG.py` กับคำถาม 5 ข้อ ดูแล้วพบ 3 อย่าง:

1. **คะแนน RRF ของ 5 อันดับแรกเป็น 0.0167, 0.0164, 0.0161, 0.0159, 0.0156 พอดี** = 1/60, 1/61, 1/62, 1/63, 1/64
   แปลว่าแต่ละ chunk ปรากฏใน ranking **แค่ list เดียว** ที่อันดับ 0–4 ติดกัน — ถ้า BM25 กับ dense เจอ chunk เดียวกัน
   คะแนนต้องสูงกว่า 1/60 (เช่น 1/60 + 1/61 = 0.0331) → **ใน top-5 ไม่มี chunk ใดได้คะแนนจากทั้งสองวิธี hybrid จึงทำงานเหมือนวิธีเดียว**
   และเพราะ context ว่าง (ข้อ 2) วิธีนั้นคือ dense — สอดคล้องกับ §4 ที่ BM25 แทบไม่เจออะไรเมื่อใช้ standard analyzer กับภาษาไทย
2. **field `context` ว่างทั้ง 25/25 ผลลัพธ์** เพราะ `rrf_fusion` เก็บ `_source` จาก ranker ตัวแรกที่เจอ chunk
   และ `anthropic-vector-index` ไม่ได้เก็บ `contextualized_content` ไว้ → chunk ที่มาจาก dense
   **ไม่มี context ส่งให้ LLM เลย** งานสร้าง context ตอน ingest จึงไม่ถูกใช้ในขั้นตอบคำถาม
3. คำตอบข้อแรกอ้าง *Morbillivirus / Paramyxoviridae* ซึ่ง **ไม่มีอยู่ใน corpus** (ตรวจด้วย grep) → LLM ตอบจากความรู้เดิม
   (prompt ของ baseline ไม่ได้สั่งให้ตอบจากเอกสารเท่านั้น) ซึ่งเป็นสิ่งที่ RAG ควรลด (สไลด์หน้า 7)

Improved แก้ทั้ง 3 ข้อ: BM25 ไทยใช้งานได้จริง (§4), index เดียวทำให้ทุก hit มี context เสมอ (§8), และ prompt ที่บังคับอ้างอิงแหล่ง (§7)

---

## §1 Chunking

**Baseline** (`baseline/authenticRAG.py` บรรทัด 67): `MarkdownNodeParser(chunk_size=256)`

- `MarkdownNodeParser` ตัดตาม **หัวข้อ markdown เท่านั้น** — พารามิเตอร์ `chunk_size` ไม่ถูกใช้
- ผลจริงบน corpus 4 ไฟล์: **53 chunks ยาว 112 – 8,404 ตัวอักษร (เฉลี่ย 1,633)**
  หัวข้อ "วิธีรักษาโรคกรดไหลย้อน" ทั้งหัวข้อ (8,404 ตัวอักษร: ยา 5 กลุ่ม, การปรับพฤติกรรม, สมุนไพร, ผ่าตัด) อยู่ใน chunk เดียว
- หัวข้อที่ใช้ `#` (H1) ซ้ำกลางเอกสาร เช่น `# วิธีป้องกันโรคหัดเยอรมัน` ทำให้ chunk ไม่มีข้อมูลว่าเป็นของโรคอะไร

**Improved** (`improved/chunking.py`)

1. แยกหัวข้อแล้วจำ **เส้นทางหัวข้อ (breadcrumb)** เช่น `หัดเยอรมัน › ภาวะแทรกซ้อนของโรคหัดเยอรมัน › ภาวะแทรกซ้อนที่พบได้ในหญิงตั้งครรภ์` และเติมชื่อโรคไว้หน้าสุดเสมอ (ย่อชื่อบทความ clickbait เหลือชื่อโรค)
2. ภายในหัวข้อ แบ่งเป็นย่อหน้า/bullet แล้ว **รวมให้ ≤ 700 ตัวอักษร** ถ้าย่อหน้าเดียวยาวเกิน ตัดที่ช่องว่าง (ขอบวลีในภาษาไทย) และถ้ายังยาวอีก ตัดที่ **ขอบคำ** ด้วย PyThaiNLP (ไม่ตัดกลางคำ)
3. **Overlap ~120 ตัวอักษร** ระหว่าง chunk ติดกัน กันประโยคสำคัญขาดตอนที่รอยต่อ
4. ตัด boilerplate: เอกสารอ้างอิง, "เรื่องที่เกี่ยวข้อง", footer ของเว็บ, URL, byline, เลขอ้างอิง `[2]`
5. ผล: **154 chunks ยาว 131 – 698 ตัวอักษร (เฉลี่ย ~510)**

**ทำไมดีขึ้น**

- *Semantic dilution*: embedding 1 เวกเตอร์ (1,024 มิติ) ต้อง "เฉลี่ย" ความหมายของทั้ง chunk ถ้า chunk มี 5 เรื่อง เวกเตอร์จะอยู่กึ่งกลางของ 5 เรื่องและไม่ใกล้กับคำถามเรื่องใดเรื่องหนึ่งจริง ๆ
- *ต้นทุน context*: ใน offline study, top-5 ของ baseline chunks ยาว 11,155 ตัวอักษร เทียบกับ 2,643 ของ improved (~4.2 เท่า) เมื่อจำกัดงบ context เท่ากันที่ 3,000 ตัวอักษร improved หาส่วนที่ตอบคำถามเจอ 87.5% vs 62.5%
- *Lost in the middle*: LLM ใช้ข้อมูลกลาง context ยาว ๆ ได้แย่กว่าข้อมูลต้น/ท้าย context สั้นและตรงประเด็นจึงลดโอกาสตอบพลาด
- Breadcrumb ทำให้ chunk เล็ก ๆ ไม่ "กำพร้า" — รู้เสมอว่าเป็นของโรคอะไร หัวข้อใด

**Trade-off**: chunk เล็กลงทำให้จำนวน chunk เพิ่ม ~3 เท่า (ค่าสร้าง context และ embedding เพิ่มตาม) และบางคำถามต้องใช้ข้อมูลข้าม chunk — จึงต้องมี §7 (ต่อ chunk ข้างเคียง) ช่วย

---

## §2 Contextual Embedding (Anthropic Contextual Retrieval)

**Baseline** (`baseline/authenticRAG.py` บรรทัด 203)

```python
full_doc_content = doc.page_content  # ถ้าไม่มีเอกสารเต็ม ใช้ chunk เอง
context = self.generate_context(full_doc_content, doc.page_content)
```

prompt ของ Anthropic คือ `<document>{เอกสารเต็ม}</document> ... <chunk>{chunk}</chunk>` แต่ baseline ส่ง **chunk เดียวกันเข้าไปทั้งสองช่อง**
LLM จึงไม่มีข้อมูลว่า chunk นี้อยู่ส่วนไหนของเอกสาร — context ที่ได้เป็นแค่ "สรุปซ้ำ" ของ chunk ไม่ใช่การ *situate* chunk ในเอกสาร ซึ่งเป็นแก่นของเทคนิค
นอกจากนี้ prompt เป็นภาษาอังกฤษล้วนและไม่ระบุภาษาของคำตอบ → **ผลจริงตอนรัน baseline: ภาษาของ context ปนกัน**
chunk 1–3 ได้ context ภาษาอังกฤษ (เช่น *"The chunk is the title and author information of a..."*, *"The chunk provides a comprehensive overview of Ger..."*)
ส่วน chunk 4 เป็นต้นไปเป็นภาษาไทยที่ **ทวนเนื้อหาของ chunk เอง** (เช่น *"อาการของโรคหัดเยอรมัน แบ่งออกเป็นการติดเชื้อในเด็ก..."*)
ไม่ได้บอกว่า chunk อยู่ส่วนไหนของเอกสาร ซึ่งตรงกับที่คาดจาก bug ข้างบน และ context ภาษาอังกฤษ BM25 จับคู่กับคำถามภาษาไทยไม่ได้
และ (ดู §0) context ที่สร้างไว้ไม่ถูกส่งให้ LLM ตอนตอบ เพราะ vector index ไม่ได้เก็บไว้

**Improved** (`improved/contextualizer.py`)

- ส่ง **เอกสารเต็มจริง** (เอกสารใน corpus ยาว ~22,000 ตัวอักษร อยู่ในงบ 24,000) ถ้ายาวเกินงบ ใช้สารบัญหัวข้อ + ข้อความรอบตำแหน่ง chunk
- ยังใช้ prompt ต้นฉบับของ Anthropic แต่เพิ่มคำสั่ง: ตอบ 1-2 ประโยค **เป็นภาษาไทย** ระบุชื่อโรค/ส่วนของเอกสาร และ **คำพ้องไทย-อังกฤษ** ที่ผู้ใช้อาจค้น
- วาง `<document>` ไว้ต้น prompt เหมือนกันทุก chunk ของเอกสารเดียวกัน (prefix คงที่ เอื้อต่อ prompt caching ของ provider)
- **Cache** ผลไว้ที่ `cache/contexts.json` (key = hash ของ prompt version + chunk) → ingest ซ้ำไม่ต้องจ่ายค่า LLM ใหม่, และทำให้ผลรันซ้ำได้ (reproducible)
- context ถูกใช้ 3 ที่: embedding, BM25 field `context_tok`, และ input ของ reranker

**ทำไมดีขึ้น**: chunk ย่อยจำนวนมากไม่มีชื่อโรคอยู่ในตัวเลย เช่น ย่อหน้าที่ขึ้นต้นว่า *"ผู้ป่วยจะมีอาการปวดแสบตรงลิ้นปี่..."* เมื่อ LLM เห็นเอกสารเต็ม จะเขียน context ทำนอง *"ส่วนอาการของโรคกรดไหลย้อน (GERD) อธิบายอาการแสบร้อนกลางอก (heartburn)"* ทำให้ทั้ง dense และ BM25 จับคู่กับคำถามที่เอ่ยชื่อโรคหรือใช้คำพ้องได้ (ดู context จริงที่ระบบสร้างได้ใน `cache/chunks.json` หลัง ingest)
ผลของ Anthropic (สไลด์หน้า 18): contextual embedding ลด failed retrieval@20 จาก 5.7% → 3.7% (−35%)

---

## §3 Embedding

| | Baseline | Improved |
|---|---|---|
| โมเดล | `BAAI/bge-m3` (1,024 มิติ) | **เหมือนเดิม** — ไม่เปลี่ยนเพื่อแยกผลของเทคนิคอื่นได้ชัด |
| ข้อความที่ embed | `chunk + context(ที่ไม่มีข้อมูลตำแหน่ง)` | `หัวข้อ: breadcrumb` + `บริบท: context` + chunk |
| Normalize | ไม่ | `normalize_embeddings=True` |
| การ encode | ทีละ chunk | batch 16 |
| Vector index | `knn_vector` ค่า default | HNSW (lucene engine, cosine, m=16, ef_construction=128) ระบุชัด |

**ทำไมดีขึ้น**: เวกเตอร์ที่ normalize แล้ว cosine = dot product คะแนนอยู่ในสเกลเดียวกันทุกเอกสาร; การใส่ breadcrumb + context ทำให้ chunk เล็ก ๆ มีสัญญาณระดับเอกสาร (โรคอะไร หัวข้ออะไร) อยู่ในเวกเตอร์ด้วย

---

## §4 Search algorithm: BM25 ภาษาไทย (การปรับปรุงที่มีผลมากที่สุด)

**Baseline** (`baseline/authenticRAG.py` บรรทัด 106-107): field ใช้ `"analyzer": "standard"`

`standard` analyzer ตัดคำตาม Unicode UAX#29 ซึ่ง **ไม่ตัดคำภาษาไทย** (ภาษาไทยไม่เว้นวรรคระหว่างคำ) ข้อความไทยที่ติดกันทั้งก้อนกลายเป็น **term เดียว** — เห็นได้ในสไลด์หน้า 24 เองที่ term คือ `"และวิธีการรักษา"` (3 คำ)

```
คำถาม: "ถ้าติดหัดเยอรมันช่วงตั้งท้องสามเดือนแรก ลูกจะเสี่ยงอะไร"
standard : ['ถ้าติดหัดเยอรมันช่วงตั้งท้องสามเดือนแรก', 'ลูกจะเสี่ยงอะไร']        ← ไม่มีทาง match เอกสาร
improved : ['ติด', 'หัดเยอรมัน', 'หัด', 'เยอรมัน', 'ตั้งท้อง', 'ท้อง', 'สาม', 'เดือน', 'ลูก', 'เสี่ยง']
```

ผลคือ BM25 ของ baseline match ได้แทบเฉพาะคำภาษาอังกฤษ → **hybrid search ของ baseline เป็น dense-only โดยปริยาย** (ข้อจำกัด "เน้น dense เกินไป" ในสไลด์หน้า 14 ถูกขยายให้หนักขึ้นอีก)

**Improved** (`improved/thai_text.py`, `improved/store.py`)

1. ตัดคำด้วย **PyThaiNLP `newmm`** ทั้งตอน index และตอน query (ฟังก์ชันเดียวกัน → token ตรงกันแน่นอน) แล้วเก็บใน field `*_tok` ที่ใช้ `whitespace` analyzer
2. **แตกคำประสม (decompound)**: dictionary มีคำยาว เช่น `โรคหัดเยอรมัน` เป็นคำเดียว ถ้าถามว่า "หัดเยอรมัน" จะไม่ match จึงเก็บทั้งคำเต็มและคำย่อย (`โรคหัดเยอรมัน, โรค, หัดเยอรมัน, หัด, เยอรมัน`) แตกเฉพาะเมื่อทุกคำย่อยเป็นคำในพจนานุกรมและยาว ≥3 ตัวอักษร (กันการแตกมั่ว เช่น `อหิวาตกโรค` → `อหิ/วา/ตก`)
3. ตัด stopword (`ที่ ของ และ อย่างไร ไหม ...`) ลด noise จากคำที่มีทุกเอกสาร
4. ค้นหลาย field แบบ **`most_fields`** (รวมคะแนนทุก field) แทน `best_fields`: `content_tok^1.0`, `breadcrumb_tok^1.5`, `context_tok^0.8` และสัญญาณเสริม `content.thai^0.3` (Lucene ThaiAnalyzer ในตัว OpenSearch)

**หลักฐาน** (offline, BM25 เท่านั้น, 24 คำถาม):

| | Recall@5 | Recall@20 |
|---|---|---|
| standard analyzer (baseline) | 4.2% | 4.2% |
| PyThaiNLP | 79.2% | 91.7% |
| PyThaiNLP + decompound (improved) | 87.5% | 100.0% |

**ทำไมสำคัญแม้มี dense อยู่แล้ว**: BM25 เก่งเรื่อง *exact match* ที่ dense มักพลาด — ชื่อยา (โอเมพราโซล), ชื่อเชื้อ (Togaviridae, วิบริโอคอเลอเร), ตัวเลข/ขนาดยา, ศัพท์เฉพาะ (Phacoemulsification) ส่วน dense เก่งเรื่องความหมายที่ใช้คำต่างกัน (ตั้งท้อง ↔ ตั้งครรภ์) สองวิธีจึงเสริมกัน — แต่ต้อง "ทำงานได้จริงทั้งคู่" ก่อน

---

## §5 Scoring / Fusion

**Baseline**: `hybrid_search(k=5)` ดึง BM25 10 + dense 10 → RRF น้ำหนักเท่ากัน → 5 อันดับแรก

**Improved** (`improved/fusion.py`):

- **Weighted RRF**: `score(d) = Σ wᵢ / (60 + rankᵢ(d))` ปรับ `WEIGHT_SPARSE`, `WEIGHT_DENSE` ได้
- **Candidate pool 30 + 30** (Anthropic ใช้ 150 → rerank → 20) เพื่อให้ reranker มีตัวเลือกมากพอ
- เก็บ **trace** ทุกขั้น (อันดับใน BM25 / dense / RRF / rerank) ไว้ในผลลัพธ์ ใช้อธิบายได้ว่าทำไม chunk นี้ถูกเลือก

**ทำไมใช้อันดับ ไม่ใช้คะแนน**: คะแนน BM25 อยู่ในช่วง 0–∞ ส่วน cosine 0–1 บวกกันตรง ๆ ไม่ได้ RRF ใช้แค่อันดับ จึงรวมได้โดยไม่ต้อง calibrate และทนต่อ outlier

---

## §6 Reranking (ของใหม่ทั้งหมด)

**Baseline**: ไม่มี — ทั้งที่สไลด์หน้า 17 ระบุว่า Reranking เป็นหลักการข้อ 2 ของ Contextual Retrieval และสไลด์หน้า 19 แสดงว่าให้ผลดีที่สุด (failed retrieval 2.9% → 1.9%)

**Improved** (`improved/models.py`, `pipeline.retrieve`)

- Cross-encoder **`BAAI/bge-reranker-v2-m3`** (หลายภาษา รองรับไทย) ให้คะแนน (คำถาม, หัวข้อ+context+chunk) ทุกตัวใน pool
- แปลงเป็นความน่าจะเป็น 0–1 แล้ว **ตัด chunk ที่คะแนน < 0.05 ทิ้ง** ไม่ส่งให้ LLM (ตรงกับสไลด์หน้า 17: "กรองเฉพาะ chunk ที่เกี่ยวข้อง")
- เหลือ top-5

**ทำไมดีขึ้น**

```
Bi-encoder (embedding):   q → [vec_q]      d → [vec_d]      sim = cos(vec_q, vec_d)
                          อ่านคำถามกับเอกสาร "แยกกัน" → เร็ว, คำนวณ vec_d ล่วงหน้าได้ แต่หยาบ
Cross-encoder (reranker): [q ; d] → transformer → score
                          ทุก token ของคำถาม attend กับทุก token ของเอกสาร → แม่นกว่ามาก แต่ช้า
```

จึงใช้แบบ 2 ขั้น: ค้นกว้างด้วยวิธีเร็ว (60 chunks) → ตัดสินละเอียดด้วย cross-encoder (เหลือ 5)
เราใช้ cross-encoder แทน "generative model" ตามสไลด์หน้า 17 เพราะ (1) ไม่ต้องเรียก LLM เพิ่มทุกคำถาม (ถูกและเร็วกว่า), (2) ได้คะแนนต่อเนื่องที่ใช้ทำ threshold ได้, และ (3) โจทย์ห้ามเปลี่ยน LLM ตัวตอบ — reranker เป็นส่วนของ retriever

---

## §7 Context assembly — ส่งอะไรให้ LLM

**Baseline** (`baseline/onlysearchAuthenticRAG.py` บรรทัด 142)

```python
words = context.split()
estimated_tokens = len(words) * 1.3
```

ภาษาไทยไม่มีช่องว่างระหว่างคำ — วัดบน corpus นี้ `split()` ได้เฉลี่ย **19.8 ตัวอักษรต่อ "คำ"** ทั้ง corpus 86,672 ตัวอักษรถูกประมาณว่าเป็นแค่ ~5,700 token ทั้งที่ tokenizer จริงนับได้มากกว่านั้นหลายเท่า → ขีดจำกัด 4,000 token แทบไม่เคยทำงาน และถ้าทำงานก็ตัดผิดที่

**Improved** (`pipeline.build_context`)

- นับ token ด้วย **tokenizer จริง** (ของ reranker) คุมงบ 3,500 token
- **ต่อ chunk ข้างเคียง** (±1 ในหัวข้อเดียวกัน) ให้ผล 3 อันดับแรก แล้วรวมเป็นก้อนเดียว ลบ overlap ซ้ำ → ได้ข้อมูลต่อเนื่องเหมือนอ่านจากเอกสาร
- ใส่ **เลขอ้างอิง [n] + หัวข้อ** ให้แต่ละแหล่ง และสั่ง LLM ให้อ้างอิง, ตอบจากเอกสารเท่านั้น, บอกตรง ๆ เมื่อไม่พบข้อมูล → ตรวจสอบย้อนกลับได้ ลด hallucination
- `temperature` ของการตอบ 0.6 → 0.3 (โมเดลเดิม แค่ลดความสุ่ม เพื่อให้ยึดเอกสารมากขึ้น)

---

## §8 Storage

**Baseline**: 2 index (`anthropic-vector-index`, `anthropic-bm25-index`) เขียนแยกกัน 2 รอบ bulk ถ้ารอบใดรอบหนึ่ง error ข้อมูลจะไม่ตรงกัน
และ vector index ไม่มี field context → chunk ที่ dense หาเจอไม่มี context ติดมา (§0 ข้อ 2)
**Improved**: index เดียว `improved-rag-index` — 1 document ต่อ chunk มีทั้ง `embedding` และ text fields → id ตรงกันเสมอ, ลบ/สร้างใหม่ได้ในคำสั่งเดียว, 1 shard เพื่อให้สถิติ IDF ของ BM25 คำนวณจากทั้ง corpus

---

## §9 การวัดผล

Baseline ไม่มีการวัดผล ทำให้บอกไม่ได้ว่าการเปลี่ยนแปลงใด "ดีขึ้นจริง" — improved จึงเพิ่ม:

- `eval/eval_set.json`: 24 คำถามที่ **ใช้คำต่างจากเอกสาร** (ตั้งท้อง/ตั้งครรภ์, โรคห่า/อหิวาตกโรค) ครอบคลุม 4 โรค แต่ละข้อมี *gold phrase* ที่ unit test ยืนยันว่ามีอยู่ใน chunk จริง
- ตัวชี้วัดเดียวกับ Anthropic: Recall@k, **Failed@k** (= 1 − Recall@k), MRR@10
- **Recall@งบตัวอักษร**: ต่อผลค้นหาตามอันดับแล้วตัดที่ 1,500 / 3,000 ตัวอักษร (จำลองงบ context ของ LLM) — จำเป็นเพราะ Recall@k ปกติ "ให้รางวัล" chunk ใหญ่โดยไม่ยุติธรรม (top-5 ของ chunk 8,000 ตัวอักษรครอบคลุมเอกสารเกือบครึ่งเล่ม)

**ข้อจำกัดของการวัดผล** (ต้องยอมรับตรง ๆ):
- eval set มี 24 ข้อ เขียนโดยผู้ทำงานเอง → มี bias ได้ ตัวเลขควรอ่านเป็น "แนวโน้ม" ไม่ใช่ค่าที่ generalize ได้
- gold phrase match แบบ substring: ถ้า phrase ถูกตัดคร่อม 2 chunk จะนับว่าไม่เจอ (unit test ตรวจแล้วว่าทุก phrase อยู่ครบใน chunk อย่างน้อย 1 ชิ้น)
- Offline study วัดเฉพาะ BM25; ผลของ context/dense/rerank ต้องรัน `eval/evaluate_retrieval.py` กับระบบจริง

---

## §10 สิ่งที่ไม่ได้เปลี่ยน และต้นทุนที่เพิ่ม

| ไม่เปลี่ยน | ต้นทุนที่เพิ่ม |
|---|---|
| LLM ตัวตอบ `qwen/qwen-2.5-72b-instruct` | เรียก LLM สร้าง context 154 ครั้ง (เทียบ 53) แต่ละครั้งส่งเอกสารเต็ม (cache แล้ว จ่ายครั้งเดียว) |
| Embedding model `BAAI/bge-m3` | โหลด reranker เพิ่ม ~2.3 GB, rerank 60 คู่ต่อคำถาม (~1–3 วินาทีบน CPU) |
| OpenSearch 2.19.1 | ตัดคำไทยตอน ingest/query (ไม่กี่มิลลิวินาทีต่อ chunk) |
| corpus เดิม 4 ไฟล์ | — |
