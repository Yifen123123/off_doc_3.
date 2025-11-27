# wrap_as_json.py
"""
將整份公文轉成：
{
  "text": "......\n......\n......"
}
讓你可以直接貼進 FastAPI 的 API body。
"""

import json
import sys


def wrap_text(raw: str) -> str:
    """
    接收原始公文內容（可含多行），輸出成可直接貼 API 的 JSON 格式。
    """
    # 保留內容不動（含換行）
    data = {"text": raw}

    # 用 json.dumps 自動處理換行 → 變成 \n
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("請貼上公文全文，貼完後按 Ctrl+D（Mac/Linux）或 Ctrl+Z 再 Enter（Windows）：\n")
    
    raw_text = sys.stdin.read()
    result = wrap_text(raw_text)
    
    print("\n======= 以下是可直接貼進 API 的內容 =======\n")
    print(result)
