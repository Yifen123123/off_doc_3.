from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROMPTS = BASE / "prompts"

def load_prompt(text: str, doc_class: str) -> str:
    # 讀 core，注入文本
    core = (PROMPTS / "core_extract.prompt").read_text(encoding="utf-8")
    core = core.replace("{{TEXT}}", text)

    # 尋找同名 add-on：prompts/addons/<類別>.prompt
    addon_path = PROMPTS / "addons" / f"{doc_class}.prompt"
    if addon_path.exists():
        print(f"[ROUTER] add-on FOUND for class '{doc_class}': {addon_path}")
        addon_header = "\n\n【以下為本類別的補充規則（同等或更高優先權）】\n"
        addon = addon_header + addon_path.read_text(encoding="utf-8")
        return core + addon
    else:
        print(f"[ROUTER] add-on MISSING for class '{doc_class}': {addon_path}")
        return core
