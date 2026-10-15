"""
ถาม-ค้น-วิเคราะห์-สรุปคำตอบ (เทียบเท่า Hands-on 2 / onlysearchAuthenticRAG.py)

    python -m improved.ask                              # ชุดคำถามเดียวกับ baseline → results/improved_answers.json
    python -m improved.ask -q "อาการของต้อกระจกเป็นอย่างไร"
"""
import argparse
import json
from pathlib import Path

from .pipeline import ImprovedRAG

# ชุดคำถามเดียวกับ onlysearchAuthenticRAG.py ของ baseline (เทียบผลได้ตรง ๆ)
DEFAULT_QUESTIONS = [
    "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
    "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
    "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
    "โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", action="append", help="ถามได้หลายข้อ: -q ... -q ...")
    ap.add_argument("-o", "--output", default="results/improved_answers.json")
    args = ap.parse_args()

    rag = ImprovedRAG()
    questions = args.question or DEFAULT_QUESTIONS
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n===== คำถามที่ {i}/{len(questions)}: {q}")
        r = rag.answer(q)
        print(f"\nคำตอบ:\n{r['answer']}")
        results.append(r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกผลลัพธ์ไปยัง {args.output}")


if __name__ == "__main__":
    main()
