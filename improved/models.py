"""
Embedding model และ Reranker (cross-encoder)

Embedding — ยังใช้ BAAI/bge-m3 เหมือน baseline แต่:
    - normalize_embeddings=True → เวกเตอร์ความยาว 1 ทำให้ cosine = dot product, คะแนนเสถียร
    - encode แบบ batch (baseline encode ทีละ chunk)
    - สิ่งที่ embed คือ "หัวข้อ + context + เนื้อหา" (baseline: เนื้อหา + context ที่ไม่มีข้อมูลตำแหน่ง)

Reranker — ของใหม่ทั้งหมด (baseline ไม่มี แม้สไลด์หน้า 17/19 จะระบุว่าเป็นขั้นที่ 2 ของ Contextual Retrieval)
    BAAI/bge-reranker-v2-m3 เป็น cross-encoder หลายภาษา (รองรับไทย)
    Bi-encoder (embedding): encode คำถามกับเอกสาร "แยกกัน" แล้วเทียบเวกเตอร์ → เร็วแต่หยาบ
    Cross-encoder: อ่าน (คำถาม, เอกสาร) "พร้อมกัน" ใน transformer เดียว ทุก token ของคำถาม
    attend กับทุก token ของเอกสาร → ตัดสินความเกี่ยวข้องได้แม่นกว่ามาก แต่ช้า
    จึงใช้แบบ 2 ขั้น: retrieve กว้าง (30-60 chunks) ด้วยวิธีเร็ว → rerank ละเอียดเหลือ top 5
"""
import math


class Embedder:
    def __init__(self, model_name: str, max_seq_len: int = 1024, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_len
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


class Reranker:
    def __init__(self, model_name: str, max_length: int = 1024):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.tokenizer = self.model.tokenizer

    def score(self, query: str, passages):
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=16,
                                 show_progress_bar=False)
        raw = [float(s) for s in raw]
        # sentence-transformers ส่วนใหญ่ใส่ sigmoid ให้แล้ว (ค่า 0-1) แต่บางเวอร์ชันคืน logit ดิบ
        # → ถ้าเจอค่าที่อยู่นอก [0,1] ค่อยแปลงด้วย sigmoid เอง (ลำดับไม่เปลี่ยน, ใช้ threshold ได้)
        if any(s < 0.0 or s > 1.0 for s in raw):
            raw = [1.0 / (1.0 + math.exp(-s)) for s in raw]
        return raw

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
