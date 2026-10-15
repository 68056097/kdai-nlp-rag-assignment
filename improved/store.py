"""
OpenSearch store: index เดียวที่มีทั้ง dense vector และ text fields สำหรับ BM25

ต่างจาก baseline:
    baseline: 2 index (anthropic-vector-index / anthropic-bm25-index) ต้องเขียนให้ตรงกันเอง
    improved: 1 index, 1 document ต่อ 1 chunk → id ตรงกันแน่นอน, ไม่มีปัญหา index ไม่ sync

Field สำคัญ
    embedding        knn_vector (HNSW, cosine) ของ passage = หัวข้อ + context + เนื้อหา
    content_tok      เนื้อหาที่ตัดคำไทยด้วย PyThaiNLP แล้ว → whitespace analyzer
    context_tok      context ที่ตัดคำแล้ว
    breadcrumb_tok   เส้นทางหัวข้อที่ตัดคำแล้ว (ให้น้ำหนักสูง เพราะหัวข้อบอกเรื่องชัดที่สุด)
    content          ข้อความดิบ + sub-field `content.thai` (Lucene ThaiAnalyzer ในตัว OpenSearch)
                     ใช้เป็นสัญญาณเสริม กรณี dictionary ของ PyThaiNLP ตัดคำต่างไป
"""
from opensearchpy import OpenSearch, helpers

from .thai_text import to_index_text


class OpenSearchStore:
    def __init__(self, host, port, index_name):
        self.client = OpenSearch(hosts=[{"host": host, "port": port}], use_ssl=False, timeout=60)
        self.index = index_name

    # ------------------------------------------------------------------ schema
    def recreate_index(self, dim: int):
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        body = {
            "settings": {
                "index.knn": True,
                "number_of_shards": 1,            # corpus เล็ก: 1 shard → สถิติ IDF ของ BM25 คำนวณจากทั้ง corpus
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}},
                "analysis": {"analyzer": {"pretok": {"type": "custom", "tokenizer": "whitespace",
                                                     "filter": ["lowercase"]}}},
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector", "dimension": dim,
                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil",
                                   "parameters": {"m": 16, "ef_construction": 128}},
                    },
                    "content": {"type": "text", "analyzer": "standard",
                                "fields": {"thai": {"type": "text", "analyzer": "thai"}}},
                    "context": {"type": "text", "analyzer": "standard"},
                    "content_tok": {"type": "text", "analyzer": "pretok"},
                    "context_tok": {"type": "text", "analyzer": "pretok"},
                    "breadcrumb_tok": {"type": "text", "analyzer": "pretok"},
                    "breadcrumb": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "section_id": {"type": "integer"},
                    "position": {"type": "integer"},
                }
            },
        }
        self.client.indices.create(index=self.index, body=body)

    # ------------------------------------------------------------------ write
    def bulk_index(self, chunks, embeddings):
        actions = []
        for ch, emb in zip(chunks, embeddings):
            actions.append({
                "_index": self.index, "_id": ch.chunk_id,
                "_source": {
                    "embedding": emb,
                    "content": ch.text, "context": ch.context,
                    "content_tok": to_index_text(ch.text),
                    "context_tok": to_index_text(ch.context),
                    "breadcrumb_tok": to_index_text(ch.breadcrumb),
                    "breadcrumb": ch.breadcrumb, "title": ch.title,
                    "doc_id": ch.doc_id, "chunk_id": ch.chunk_id, "source": ch.source,
                    "section_id": ch.section_id, "position": ch.position,
                },
            })
        ok, errors = helpers.bulk(self.client, actions, raise_on_error=False)
        self.client.indices.refresh(index=self.index)
        return ok, errors

    # ------------------------------------------------------------------ read
    _SOURCE = {"excludes": ["embedding", "content_tok", "context_tok", "breadcrumb_tok"]}

    def sparse_search(self, query: str, k: int, use_context: bool = True):
        q_tok = to_index_text(query)
        fields = ["content_tok^1.0", "breadcrumb_tok^1.5", "content.thai^0.3"]
        if use_context:
            fields.append("context_tok^0.8")
        body = {
            "size": k, "_source": self._SOURCE,
            "query": {"multi_match": {
                # most_fields: รวมคะแนนจากหลาย field (คำที่อยู่ทั้งหัวข้อและเนื้อหาได้คะแนนสะสม)
                # ต่างจาก best_fields ของ baseline ที่เอาเฉพาะ field ที่ดีที่สุด
                "query": q_tok or query, "fields": fields, "type": "most_fields",
            }},
        }
        res = self.client.search(index=self.index, body=body)
        return [(h["_id"], h["_score"], h["_source"]) for h in res["hits"]["hits"]]

    def dense_search(self, query_vec, k: int):
        body = {"size": k, "_source": self._SOURCE,
                "query": {"knn": {"embedding": {"vector": query_vec, "k": k}}}}
        res = self.client.search(index=self.index, body=body)
        return [(h["_id"], h["_score"], h["_source"]) for h in res["hits"]["hits"]]

    def get_by_positions(self, doc_id: str, positions):
        body = {"size": len(positions), "_source": self._SOURCE,
                "query": {"bool": {"filter": [{"term": {"doc_id": doc_id}},
                                              {"terms": {"position": list(positions)}}]}}}
        res = self.client.search(index=self.index, body=body)
        return {h["_source"]["position"]: h["_source"] for h in res["hits"]["hits"]}

    def count(self):
        return self.client.count(index=self.index)["count"]
