import json, urllib.request, urllib.error, sys

URL  = "http://YOUR_OLLAMA_HOST:11434/api/pull"  # 改成你的主機
NAME = "qwen2.5:7b-instruct"                      # 要下載的模型:tag

def main():
    data = json.dumps({"name": NAME}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        # /api/pull 會串流 NDJSON；這裡逐行讀取並即時顯示進度
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line: 
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    print(line); continue
                status = obj.get("status") or ""
                completed = obj.get("completed")
                total = obj.get("total")
                if completed is not None and total:
                    print(f"{status} {completed}/{total}", flush=True)
                else:
                    print(status or line, flush=True)
        print("[done]")
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}] {e.reason}\n{e.read().decode('utf-8',errors='replace')}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr); sys.exit(2)

if __name__ == "__main__":
    main()
