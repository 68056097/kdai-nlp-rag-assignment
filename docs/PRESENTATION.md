# (6) โน้ตนำเสนอในห้อง (~7 นาที)

## สไลด์/หัวข้อที่ 1 — โจทย์ (30 วินาที)
"ดิฉันทำ Authentic RAG ตามสไลด์หน้า 22, 25 เป็น baseline แล้วปรับเฉพาะส่วน retriever — chunking, ตัดคำ, embedding, fusion, reranking — โดย **ไม่เปลี่ยน Qwen ตัวตอบ** เพราะสไลด์หน้า 15 บอกว่าความสำเร็จของ RAG เริ่มที่ retriever"

## หัวข้อที่ 2 — เจอปัญหาอะไรใน baseline (2 นาที) ← จุดที่น่าสนใจที่สุด
เปิด `results/baseline_reference_output_from_instructor_repo.json` ให้เพื่อนดู:
1. คะแนน RRF = 0.0167, 0.0164, 0.0161 … = 1/60, 1/61, 1/62 → ไม่มี chunk ใดถูกเจอโดยทั้งสองวิธี "hybrid" จริง ๆ ทำงานวิธีเดียว
2. `context` ว่างทุกผลลัพธ์ → context ที่จ่ายเงินสร้างตอน ingest ไม่เคยไปถึง LLM
3. ต้นเหตุหลัก: `standard` analyzer ไม่ตัดคำไทย — ชี้สไลด์หน้า 24 ที่ term คือ "และวิธีการรักษา" (3 คำรวมเป็น 1)
   - โชว์: `ถ้าติดหัดเยอรมันช่วงตั้งท้องสามเดือนแรก` → standard ได้ 1 token / PyThaiNLP ได้ 7+ token
4. bug เล็กแต่สำคัญ: `full_doc_content = doc.page_content` → ส่ง chunk แทนเอกสารเต็มให้ LLM สร้าง context
5. `MarkdownNodeParser(chunk_size=256)` — ค่า 256 ไม่มีผล chunk ยาวถึง 8,404 ตัวอักษร

## หัวข้อที่ 3 — แก้อย่างไร (2 นาที)
เปิดแผนภาพใน README ส่วนที่ 1 ไล่จากซ้ายไปขวา:
chunk ตามโครงสร้าง+breadcrumb → context จากเอกสารเต็ม (ภาษาไทย, cache) → embed หัวข้อ+context+chunk → BM25 ไทย (PyThaiNLP + แตกคำประสม) → Weighted RRF pool 30+30 → cross-encoder rerank + ตัดที่ไม่เกี่ยวข้อง → ต่อ chunk ข้างเคียง + อ้างอิง [n]

อธิบายต่าง bi-encoder vs cross-encoder ด้วยภาพเดียว (COMPARISON §6)

## หัวข้อที่ 4 — ผลลัพธ์ (1.5 นาที)
เปิด README ส่วน 3.2 (ผลจริงบน OpenSearch, ไฟล์เต็ม `results/retrieval_eval_noctx.md`):
- **Baseline Hybrid = Baseline Dense ทุกตัวเลข** → BM25 ของ baseline ไม่ได้ช่วยอะไรใน fusion
- BM25 ภาษาไทย Recall@5 **4.2% → 83.3%** แค่เปลี่ยนวิธีตัดคำ
- ระบบเต็ม MRR **0.924 → 0.965** ทั้งที่ส่ง context ให้ LLM น้อยลง **~4 เท่า** (11,093 → 2,759 ตัวอักษร)
- จุดที่อยากให้เพื่อนสังเกต: **Recall@k หลอกได้** — baseline chunk ใหญ่จึงเจอคำตอบง่าย พอคุมงบให้ LLM อ่านได้ 1,500 ตัวอักษรเท่ากัน baseline เหลือ 75.0% ส่วน improved 95.8%
- **Hybrid ไม่ได้ดีกว่าเสมอ**: RRF น้ำหนักเท่ากันทำ Recall@1 ตกเหลือ 66.7% → reranker แก้กลับเป็น 95.8% (เช่น q22 อันดับ 9 → 1)
- ข้อที่ยังพลาด q01 ("ตั้งท้องสามเดือนแรก" vs เอกสารเขียน "ไตรมาสแรกของการตั้งครรภ์") → กรณีที่ contextual chunk ออกแบบมาแก้
- ผลรอบนี้ปิดการสร้าง context ไว้ (เครดิต API ไม่พอ) — โค้ดพร้อมแล้ว รันเพิ่มได้

## หัวข้อที่ 5 — ข้อจำกัด/สิ่งที่จะทำต่อ (30 วินาที)
- eval set 24 ข้อเขียนเอง → ควรให้คนอื่นเขียนคำถามเพิ่ม
- ต้นทุน: เรียก LLM สร้าง context 154 ครั้ง (cache แล้ว), reranker ใช้ RAM เพิ่ม
- ต่อยอด: ใช้ sparse vector ของ bge-m3 แทน BM25, query rewriting, ประเมินคำตอบด้วย RAGAS (faithfulness)

## คำถามที่อาจโดนถาม
- **ทำไมไม่ใช้ LLM rerank ตามสไลด์หน้า 17?** → cross-encoder ถูกกว่า/เร็วกว่า ให้คะแนนต่อเนื่องที่ทำ threshold ได้ และไม่ต้องเรียก LLM เพิ่มทุกคำถาม (โจทย์ให้คง LLM ตัวตอบไว้)
- **ทำไมไม่ใช้ `thai` analyzer ของ OpenSearch อย่างเดียว?** → ใช้เป็นสัญญาณเสริม (`content.thai`) แต่ PyThaiNLP ควบคุมได้มากกว่า (stopword, แตกคำประสม) และใช้ฟังก์ชันเดียวกันทั้งตอน index/query
- **chunk 700 ตัวอักษรมาจากไหน?** → ประมาณ 1–2 ย่อหน้าของเอกสาร medthai ปรับได้ผ่าน `CHUNK_MAX_CHARS` แล้ววัดด้วย eval
