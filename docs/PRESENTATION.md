# (6) โน้ตนำเสนอในห้อง (~7 นาที)

## สไลด์/หัวข้อที่ 1 — โจทย์ (30 วินาที)
"ผมทำ Authentic RAG ตามสไลด์หน้า 22, 25 เป็น baseline แล้วปรับเฉพาะส่วน retriever — chunking, ตัดคำ, embedding, fusion, reranking — โดย **ไม่เปลี่ยน Qwen ตัวตอบ** เพราะสไลด์หน้า 15 บอกว่าความสำเร็จของ RAG เริ่มที่ retriever"

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
- ตาราง offline: BM25 Recall@5 จาก **4.2% → 87.5%** แค่เปลี่ยนวิธีตัดคำ + chunking
- จุดที่อยากให้เพื่อนสังเกต: **Recall@k หลอกได้** — chunk ใหญ่ได้ Recall@5 = 100% แต่ top-5 ยาว 11,155 ตัวอักษร
  พอคุมงบ context เท่ากัน 3,000 ตัวอักษร chunk เล็กชนะ 87.5% vs 62.5% → ต้องวัดผลให้ยุติธรรม
- ตารางระบบจริง (`results/retrieval_eval.md`) และคำตอบเคียงข้างกัน (`results/answers_side_by_side.md`)

## หัวข้อที่ 5 — ข้อจำกัด/สิ่งที่จะทำต่อ (30 วินาที)
- eval set 24 ข้อเขียนเอง → ควรให้คนอื่นเขียนคำถามเพิ่ม
- ต้นทุน: เรียก LLM สร้าง context 154 ครั้ง (cache แล้ว), reranker ใช้ RAM เพิ่ม
- ต่อยอด: ใช้ sparse vector ของ bge-m3 แทน BM25, query rewriting, ประเมินคำตอบด้วย RAGAS (faithfulness)

## คำถามที่อาจโดนถาม
- **ทำไมไม่ใช้ LLM rerank ตามสไลด์หน้า 17?** → cross-encoder ถูกกว่า/เร็วกว่า ให้คะแนนต่อเนื่องที่ทำ threshold ได้ และไม่ต้องเรียก LLM เพิ่มทุกคำถาม (โจทย์ให้คง LLM ตัวตอบไว้)
- **ทำไมไม่ใช้ `thai` analyzer ของ OpenSearch อย่างเดียว?** → ใช้เป็นสัญญาณเสริม (`content.thai`) แต่ PyThaiNLP ควบคุมได้มากกว่า (stopword, แตกคำประสม) และใช้ฟังก์ชันเดียวกันทั้งตอน index/query
- **chunk 700 ตัวอักษรมาจากไหน?** → ประมาณ 1–2 ย่อหน้าของเอกสาร medthai ปรับได้ผ่าน `CHUNK_MAX_CHARS` แล้ววัดด้วย eval
