import os
import glob
import json
import pathlib
from typing import Dict, Any

import ollama
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# =====================[ CONFIG ]=====================
OCR_DIR      = "uploaded_docs"
PROMPTS_DIR  = "prompts"
OUTPUT_DIR   = "outputs"
MODEL        = "qwen2.5:3b-instruct"
SYSTEM_PROMPT_CORE = (
    "你是法務抽取器。請依照模板指示，從 OCR 文字中抽取『核心欄位』。"
    "若資料不足請以空字串或 null 填入，勿臆測。盡量輸出嚴格的 JSON。"
)
SYSTEM_PROMPT_FINAL = (
    "你是法務助理。請依模板與給定的核心欄位（core）及原始 OCR 文本生成最終輸出。"
    "若資料不足請明確標註，勿臆測。"
)

# 指定順序：第一階段一定是 core_extract，第二階段是 扣押命令
CORE_TEMPLATE_NAME  = "core_extract.prompt"
FINAL_TEMPLATE_NAME = "扣押命令.prompt"

# 保留階段性輸出檔案（各自 .out.txt）
SAVE_STAGE_FILES = True

# 讀 OLLAMA 主機（可環境變數覆寫）
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# 若模板需要其他共用變數，可放這裡
EXTRA_CONTEXT: Dict[str, Any] = {
    # "today": "2025-11-07"
}

# =====================[ 工具函式 ]=====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_ocr_texts(ocr_dir: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(ocr_dir, "*.txt"))):
        try:
            txt = pathlib.Path(p).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = pathlib.Path(p).read_text(encoding="cp950", errors="ignore")
        data[os.path.basename(p)] = txt.strip()
    return data

def build_jinja_env(prompts_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(prompts_dir, encoding="utf-8"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

def render(env: Environment, template_name: str, **ctx) -> str:
    tmpl = env.get_template(template_name)
    return tmpl.render(**ctx)

def call_ollama(host: str, model: str, system_prompt: str, user_prompt: str) -> str:
    client = ollama.Client(host=host)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    return resp["message"]["content"]

def try_parse_json(text: str) -> Any:
    # 嘗試從模型輸出中抓純 JSON
    stripped = text.strip()
    # 去掉可能包在 ```json ... ``` 的格式
    if stripped.startswith("```"):
        # 粗略剝殼
        lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except Exception:
        return None

# =====================[ 主流程 ]=====================
def main():
    if not os.path.isdir(OCR_DIR):
        raise SystemExit(f"❌ 找不到資料夾：{OCR_DIR}/")
    if not os.path.isdir(PROMPTS_DIR):
        raise SystemExit(f"❌ 找不到資料夾：{PROMPTS_DIR}/")
    ensure_dir(OUTPUT_DIR)

    ocr_map = load_ocr_texts(OCR_DIR)
    if not ocr_map:
        raise SystemExit("❌ 沒有 OCR .txt 檔案")

    env = build_jinja_env(PROMPTS_DIR)

    print(f"✅ OCR 檔案：{len(ocr_map)} 份")
    print(f"➡️  Ollama：{OLLAMA_HOST}，模型：{MODEL}")
    print(f"➡️  輸出：{OUTPUT_DIR}/\n")

    for fname, ocr_text in ocr_map.items():
        base = os.path.splitext(fname)[0]

        # ---------- 第 1 階段：core_extract ----------
        try:
            user_prompt_core = render(
                env, CORE_TEMPLATE_NAME,
                ocr_text=ocr_text,
                **EXTRA_CONTEXT
            )
        except Exception as e:
            print(f"⚠️ 模板渲染失敗（{CORE_TEMPLATE_NAME} × {fname}）：{e}")
            continue

        try:
            core_answer = call_ollama(
                host=OLLAMA_HOST,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT_CORE,
                user_prompt=user_prompt_core
            )
        except Exception as e:
            print(f"❌ 模型呼叫失敗（{CORE_TEMPLATE_NAME} × {fname}）：{e}")
            continue

        # 儲存階段一輸出（可關閉）
        if SAVE_STAGE_FILES:
            out_core = os.path.join(OUTPUT_DIR, f"{base}__core_extract.prompt.out.txt")
            pathlib.Path(out_core).write_text(core_answer, encoding="utf-8")
            print(f"📝 已產出：{os.path.basename(out_core)}")

        # 嘗試解析 JSON，供第二階段引用（core）
        core_obj = try_parse_json(core_answer)
        if core_obj is None:
            # 若非 JSON，仍放進 core_raw 讓第 2 階段可引用原文
            core_ctx = {"core": {}, "core_raw": core_answer}
        else:
            core_ctx = {"core": core_obj, "core_raw": core_answer}

        # ---------- 第 2 階段：扣押命令 ----------
        try:
            user_prompt_final = render(
                env, FINAL_TEMPLATE_NAME,
                ocr_text=ocr_text,
                **EXTRA_CONTEXT,
                **core_ctx
            )
        except Exception as e:
            print(f"⚠️ 模板渲染失敗（{FINAL_TEMPLATE_NAME} × {fname}）：{e}")
            continue

        try:
            final_answer = call_ollama(
                host=OLLAMA_HOST,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT_FINAL,
                user_prompt=user_prompt_final
            )
        except Exception as e:
            print(f"❌ 模型呼叫失敗（{FINAL_TEMPLATE_NAME} × {fname}）：{e}")
            continue

        # 儲存階段二輸出（可關閉）
        if SAVE_STAGE_FILES:
            out_phase2 = os.path.join(OUTPUT_DIR, f"{base}__扣押命令.prompt.out.txt")
            pathlib.Path(out_phase2).write_text(final_answer, encoding="utf-8")
            print(f"📝 已產出：{os.path.basename(out_phase2)}")

        # ---------- 合併最終輸出 ----------
        final_path = os.path.join(OUTPUT_DIR, f"{base}__final.out.txt")
        merged = [
            "=== core_extract（第1階段）===\n",
            core_answer.strip(), "\n\n",
            "=== 扣押命令（第2階段）===\n",
            final_answer.strip(), "\n"
        ]
        pathlib.Path(final_path).write_text("".join(merged), encoding="utf-8")
        print(f"✅ 最終完成：{os.path.basename(final_path)}\n")

    print("🎉 全部完成。")

if __name__ == "__main__":
    main()
