# ollama_native.py
import json, urllib.request, urllib.error, sys

URL   = "http://YOUR_OLLAMA_HOST:11434/api/chat"  # 把這行改成你的完整端點
MODEL = "qwen2.5:7b-instruct"
PROMPT = "why is the sky blue?"

def main():
    payload = {"model": MODEL, "messages": [{"role":"user","content":PROMPT}], "stream": False}
    req = urllib.request.Request(URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
            print((obj.get("message") or {}).get("content", obj.get("response","")))
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}] {e.reason}\n{e.read().decode('utf-8',errors='replace')}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr); sys.exit(2)

if __name__ == "__main__":
    main()
