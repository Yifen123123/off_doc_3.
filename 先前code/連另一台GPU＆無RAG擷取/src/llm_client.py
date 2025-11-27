# src/llm_client.py
import os, json, requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def _extract_json_object(text: str) -> str | None:
    """
    從任意文字裡抓出「第一個完整且括號平衡」的 JSON 物件字串。
    會避開字串中的大括號，並處理跳脫字元。
    """
    in_str = False
    esc = False
    start = -1
    depth = 0
    best = None

    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str
        if in_str:
            esc = (ch == '\\' and not esc)
            continue
        esc = False

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = text[start:i+1].strip()
                    # 拿第一個完整物件；若有多個，取最長者
                    if best is None or len(cand) > len(best):
                        best = cand
                    start = -1
    return best

def chat_ollama_raw(prompt: str, model: str, temperature: float = 0.2, timeout: int = 600) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,                 # 避免串流被中間設備切斷
        "format": "json",                # ★ 要求模型輸出 JSON
        "options": {"temperature": temperature}
    }
    headers = {"Connection": "close"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # 有些模型在 format=json 下會把 JSON 放在 message["content"]
    return data["message"]["content"]

def chat_ollama_json(prompt: str, model: str, temperature: float = 0.2) -> dict:
    text = chat_ollama_raw(prompt, model=model, temperature=temperature)

    # 直接嘗試
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 剪掉 code fences
    if "```" in text:
        # 優先找 ```json ... ``` 區塊
        parts = []
        blocks = text.split("```")
        for b in blocks:
            if b.lstrip().lower().startswith("json"):
                parts.append(b.split("\n", 1)[1] if "\n" in b else "")
            else:
                parts.append(b)
        text = max(parts, key=len).strip()

    # 用括號平衡抓最大 JSON 物件
    cand = _extract_json_object(text)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            # 再試一次：去掉常見尾端逗號
            cand2 = cand.replace(",}", "}").replace(",]", "]")
            return json.loads(cand2)

    # 到這裡仍失敗，把原文寫到 debug 再拋出
    try:
        import uuid, pathlib
        p = pathlib.Path("outputs/_raw_llm_" + str(uuid.uuid4()) + ".txt")
        p.write_text(text, encoding="utf-8")
    except Exception:
        pass
    # 最後拋出原始錯誤
    return json.loads(text)  # 讓原本錯誤冒出來以利定位
