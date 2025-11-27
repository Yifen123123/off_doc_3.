import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any

from .router import load_prompt
from .llm_client import chat_ollama_json
from .utils import (
    validate_tw_id, today_fields,
    extract_officer, extract_phone, extract_doc_no,  # 你原本就有
    extract_doc_date                                 # ← 新增這個
)

BASE = Path(__file__).resolve().parents[1]

def post_validate(payload: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    # 1) 身分證檢核（僅驗證，不主動新增 targets）
    targets = payload.get("targets", []) or []
    for p in targets:
        twid = p.get("tw_id")
        p["id_valid"] = bool(twid and validate_tw_id(twid))
    payload["targets"] = targets

    # 2) officer/phone/doc_no 的 fallback（保持你現有的輔助規則）
    if not payload.get("officer_role") and not payload.get("officer_name"):
        role, name = extract_officer(raw_text)
        payload["officer_role"] = role
        payload["officer_name"] = name

    if not payload.get("contact_phone"):
        phone = extract_phone(raw_text)
        if phone:
            payload["contact_phone"] = phone

    if not payload.get("doc_no"):
        doc_no = extract_doc_no(raw_text)
        if doc_no:
            payload["doc_no"] = doc_no

    if not payload.get("doc_date"):
        dd = extract_doc_date(raw_text)
        if dd:
            payload["doc_date"] = dd

    return payload

def render_reply(payload: Dict[str, Any]) -> str:
    from datetime import datetime
    env = Environment(loader=FileSystemLoader(str(BASE / "templates")), trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template("reply_letter.txt.j2")

    # 日期顯示：優先 doc_date，沒有才用今天
    today = today_fields()
    doc_date_iso = payload.get("doc_date") or today["today_iso"]
    try:
        dt = datetime.fromisoformat(doc_date_iso)
        display = {
            "display_yyyy": f"{dt.year}",
            "display_mm": f"{dt.month:02d}",
            "display_dd": f"{dt.day:02d}",
            "display_iso": dt.strftime("%Y-%m-%d"),
        }
    except Exception:
        display = {
            "display_yyyy": today["today_yyyy"],
            "display_mm": today["today_mm"],
            "display_dd": today["today_dd"],
            "display_iso": today["today_iso"],
        }

    text = tpl.render(
        agency=payload.get("agency"),
        reference_date=payload.get("reference_date"),
        doc_no=payload.get("doc_no"),
        officer_role=payload.get("officer_role"),
        officer_name=payload.get("officer_name"),
        contact_phone=payload.get("contact_phone"),

        targets=payload.get("targets", []),

        # ★ 關鍵：把 add-on 寫入的內容帶進模板
        policies=payload.get("policies", []),
        class_specific=payload.get("class_specific", {}),

        display_yyyy=display["display_yyyy"],
        display_mm=display["display_mm"],
        display_dd=display["display_dd"],
        display_iso=display["display_iso"],
    )
    return text


def infer_class_from_path(p: Path) -> str:
    # 假設你的資料結構為 data/<類別>/<檔名>.txt
    return p.parent.name

def main():
    ap = argparse.ArgumentParser(description="Gov letter router → LLM extract → reply text")
    ap.add_argument("--data-dir", default=str(BASE / "data"), help="資料夾路徑（內含已分類好的 .txt 檔）")
    ap.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama 模型名稱")
    ap.add_argument("--out-dir", default=str(BASE / "outputs"), help="輸出回文純文字的資料夾")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(data_dir.glob("*/*.txt"))
    if not txt_files:
        print(f"[WARN] 在 {data_dir} 底下未找到任何 txt（預期結構：data/<類別>/<檔名>.txt）")
        return

    for fp in txt_files:
        doc_class = infer_class_from_path(fp)
        raw_text = fp.read_text(encoding="utf-8", errors="ignore")
        prompt = load_prompt(raw_text, doc_class=doc_class)

        print(f"[INFO] 處理：{fp.name}（類別：{doc_class}）… 送 LLM 抽取")
        payload = chat_ollama_json(prompt, model=args.model, temperature=0.1)

        payload = post_validate(payload, raw_text=raw_text)
        reply = render_reply(payload)

        out_path = out_dir / (fp.stem + ".reply.txt")
        out_path.write_text(reply, encoding="utf-8")
        print(f"[OK] 已輸出：{out_path}")

if __name__ == "__main__":
    main()
