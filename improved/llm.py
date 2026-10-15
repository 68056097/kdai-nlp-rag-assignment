"""LLM client — ใช้โมเดลเดียวกับ baseline (qwen/qwen-2.5-72b-instruct ผ่าน OpenRouter)"""
import os
import time

from openai import OpenAI

from .config import SETTINGS


class QwenClient:
    def __init__(self, settings=SETTINGS):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model

    def chat(self, messages, max_tokens=500, temperature=0.2, retries=3):
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:          # rate limit / network → retry แบบ backoff
                last_err = e
                status = getattr(e, "status_code", None)
                if status in (402, 429) and attempt < retries - 1:
                    # OpenRouter บอกเวลาที่ควรรอใน Retry-After (เช่น in-flight budget เต็ม)
                    wait = 30
                    try:
                        wait = int(e.response.headers.get("Retry-After", wait))
                    except Exception:
                        pass
                    print(f"[llm] HTTP {status}: รอ {min(wait, 130)} วินาทีแล้วลองใหม่...")
                    time.sleep(min(wait, 130))
                else:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")
