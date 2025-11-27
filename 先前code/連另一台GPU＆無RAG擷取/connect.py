import requests, json, re

HOST = "http://YOUR_OLLAMA_HOST:11434"
MODEL = "qwen2.5:14b-instruct"  # 若VRAM有限改 7b

schema = {
  "type": "object",
  "properties": {
    "doc_type": {"type": "string", "enum": ["保單查詢","保單註記","扣押命令","撤銷令","通知函","其他"]},
    "title": {"type": "string"},
    "agency": {"type": "string"},
    "case_id": {"type": "string"},
    "person_name": {"type": "string"},
    "twid": {"type": "string", "pattern": "^[A-Z][12][0-9]{8}$"},
    "phone": {"type": "string"},
    "date": {"type": "string"},  # 之後你可要求 YYYY-MM-DD
    "summary": {"type": "string"}
  },
  "required": ["doc_type","title","date"],
  "additionalProperties": False
}

ocr_text = """（這裡放 OCR 後的文字）"""

system_prompt = f"""
你是資料抽取引擎。請僅輸出符合此 JSON Schema 的 JSON：
{json.dumps(schema, ensure_ascii=False, indent=2)}
規則：
- 只輸出 JSON，不得出現任何多餘字元（包含說明文字）。
- 空缺請以空字串 "" 表示；不得輸出 null。
- 若無法確定 doc_type，請填 "其他"。
- twid 必須符合正則，否則輸出 ""。
"""

payload = {
    "model": MODEL,
    "format": "json",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"以下為公文全文，請抽取欄位：\n\n{ocr_text}"}
    ],
    "stream": False,
}

r = requests.post(f"{HOST}/api/chat", json=payload, timeout=120)
r.raise_for_status()
raw = r.json()["message"]["content"]

# 解析與基本驗證
obj = json.loads(raw)

# 你可以再做第二層「確定性驗證」
def valid_twid(x):
    return bool(re.match(r"^[A-Z][12][0-9]{8}$", x or ""))

if not valid_twid(obj.get("twid","")):
    obj["twid"] = ""  # 或觸發修復/回問流程

print(obj)
