import os, json, re
import gradio as gr
from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama


# 基本設定

BASE_DIR = "rag_docs"
subdirs = ["cases", "cocepts", "laws", "structure"]  # 跟你原本一致
STRUCTURE_DIR = os.path.join(BASE_DIR, "structure")
CHROMA_DIR = "rag_chroma"

OLLAMA_MODEL = "qwen2.5:14b-instruct"  
llm = ChatOllama(model=OLLAMA_MODEL,temperature=0,)                                                 

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


# 建立 Chroma 向量庫

documents = []
for sub in subdirs:
    folder = os.path.join(BASE_DIR, sub)
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            docs = TextLoader(path, encoding="utf-8").load()
            for d in docs:
                d.metadata["category"] = sub
                d.metadata["filename"] = fname
            documents.extend(docs)

split_docs = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=200
).split_documents(documents)

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
)

def retrieve_context(query: str, k: int = 6, category=None) -> str:
    if category:
        docs = vectordb.similarity_search(query, k=k, filter={"category": category})
    else:
        docs = vectordb.similarity_search(query, k=k)

    return "\n\n".join(
        f"[{d.metadata.get('category','')}/{d.metadata.get('filename','')}] {d.page_content}"
        for d in docs
    )



# 第二部分：structure 對應

DOC_TYPE_TO_STRUCTURE = {
    "扣押命令": "扣押命令_structure.txt",
    "保單查詢": "保單查詢_structure.txt",
    "保單註記": "保單註記_structure.txt",
    "保單查詢＋註記": "保單查詢＋註記_structure.txt",
    "收取令": "收取令_structure.txt",
    "撤銷令": "撤銷令_structure.txt",
    "收取＋撤銷": "收取＋撤銷_structure.txt",
    "通知函": "通知函_structure.txt",
    "公職查詢": "公職查詢_structure.txt",
}


def load_detail_structure_text(doc_type: str) -> str:
    fname = DOC_TYPE_TO_DETAIL_STRUCTURE[doc_type]
    path = os.path.join(STRUCTURE_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_parse_json(text: str):
    t = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except:
        return {"raw_output": t}



# Part 1：基本四欄位（LLM + 規則）

SYSTEM_PROMPT_BASIC = """
你是一個台灣法院/機關公文的「基本欄位擷取助手」。
只擷取：
- 基準日
- 來函機關
- 收文編號（純數字、在前半部、不能是電話/傳真/保單）
- 查詢對象（姓名＋身分證字號，輸出格式：姓名, 身分證）
若找不到「日期行 + 鄰近純數字收文號」配對，基準日與收文編號都填「無明確記載」。
嚴格輸出 JSON，不要多餘文字。
""".strip()

DATE_PATTERNS = [
    r"(?:中華民國)?\s*\d{2,4}年\d{1,2}月\d{1,2}日",
    r"\b\d{2,4}[./-]\d{1,2}[./-]\d{1,2}\b",
]

BANNED_KEYWORDS = [
    "電話","TEL","Tel","tel","傳真","FAX","Fax","fax",
    "保單","保險","保單號","保單編號","帳號","帳戶","序號",
    "聯絡","手機","市話","統一編號","統編"
]

def _is_date_line(line: str) -> bool:
    return any(re.search(p, line.strip()) for p in DATE_PATTERNS)

def _is_pure_number_line(line: str) -> bool:
    s = line.strip().replace(" ", "")
    return s.isdigit() and len(s) >= 5

def _has_banned_keyword(line: str) -> bool:
    return any(k in line for k in BANNED_KEYWORDS)

def find_date_and_receipt_pair(raw_doc: str):
    lines = [ln.strip() for ln in raw_doc.splitlines() if ln.strip()]
    if not lines:
        return None, None
    head_lines = lines[: max(1, len(lines)//2)]

    for i, line in enumerate(head_lines):
        if not _is_date_line(line):
            continue
        for j in (i+1, i+2):
            if j >= len(head_lines):
                continue
            cand = head_lines[j]
            if _has_banned_keyword(cand):
                continue
            if _is_pure_number_line(cand):
                return line, cand
    return None, None

def normalize_target_field(target: str) -> str:
    id_match = re.search(r"[A-Z][0-9]{9}", target, re.I)
    id_number = id_match.group(0).upper() if id_match else None
    name_match = re.search(r"[\u4e00-\u9fa5]{2,4}", target)
    name = name_match.group(0) if name_match else None
    if name and id_number:
        return f"{name}, {id_number}"
    return target

def extract_basic_info_with_llm(raw_doc: str) -> dict:
    prompt = f"""
{SYSTEM_PROMPT_BASIC}

【公文原文】
{raw_doc}

請回傳 JSON：
{{
  "基準日": "",
  "來函機關": "",
  "收文編號": "",
  "查詢對象": ""
}}
""".strip()

    resp = llm.invoke(prompt).content.strip()
    cleaned = resp.replace("```json","").replace("```","").strip()
    data = json.loads(cleaned)

    dline, rno = find_date_and_receipt_pair(raw_doc)
    if dline and rno:
        data["基準日"] = dline
        data["收文編號"] = rno
    else:
        data["基準日"] = "無明確記載"
        data["收文編號"] = "無明確記載"

    if isinstance(data.get("查詢對象"), str):
        data["查詢對象"] = normalize_target_field(data["查詢對象"])

    return data



# Part 2：說明欄位（RAG + detail structure）

SYSTEM_PROMPT_DETAIL = """
你是一個專門處理台灣法院與機關公文的「欄位擷取助手」。
你的任務：
1. 讀懂公文。
2. 結合提供的法條、概念、案例說明（RAG context）。
3. 嚴格按照「輸出格式要求」產出結果。
4. 不要多加任何解釋、前言或後記，只輸出擷取結果本身。
""".strip()


def extract_with_rag(raw_doc: str, doc_type: str, k: int = 8) -> str:
    structure_text = load_structure_text(doc_type)
    rag_context = retrieve_context(query=raw_doc, k=k, category=None)

    user_prompt = f"""
公文類型：{doc_type}

【公文原文】
{raw_doc}

【相關知識（法律、概念、案例等，僅供你參考推理）】
{rag_context}

【輸出格式與欄位說明（JSON schema）】
{structure_text}

請依照 schema 輸出純 JSON，只能包含 schema 裡的 keys。
若無明確記載填「無明確記載」。
""".strip()

    return llm.invoke(user_prompt).content.strip()


# 你要刪掉的「基本欄位」keys
BASIC_LIKE_KEYS = {
    "基準日", "發文日期",
    "來函機關", "發文機關",
    "收文編號",
    "查詢對象",
    "來函資訊", "承辦資訊"
}


def prune_basic_from_detail(obj):
    """遞迴刪掉 details 裡的基本欄位"""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k in BASIC_LIKE_KEYS:
                continue
            new_obj[k] = prune_basic_from_detail(v)
        return new_obj
    elif isinstance(obj, list):
        return [prune_basic_from_detail(x) for x in obj]
    else:
        return obj


def extract_all_fields(raw_doc: str, doc_type: str):
    basic = extract_basic_info_with_llm(raw_doc)

    detail_text = extract_with_rag(raw_doc, doc_type)
    detail_json = safe_parse_json(detail_text)
    detail_clean = prune_basic_from_detail(detail_json)

    return basic, detail_clean


# # Gradio UI

# DOC_TYPES = list(DOC_TYPE_TO_STRUCTURE.keys())

# def ui_extract(doc_text, doc_type):
#     if not doc_text.strip():
#         return {"error": "請先貼上完整公文"}, {"error": "請先貼上完整公文"}

#     return extract_all_fields(doc_text, doc_type)


# with gr.Blocks() as demo:
#     gr.Markdown("# 📄 九類公文擷取（本地 Ollama + 原 structure + prune 基本欄位）")

#     doc_type = gr.Dropdown(choices=DOC_TYPES, value="扣押命令", label="公文類型")
#     doc_input = gr.Textbox(lines=18, label="請貼上完整公文原文（OCR文字）")
#     btn = gr.Button("開始擷取")

#     with gr.Tabs():
#         with gr.Tab("基本欄位（四項）"):
#             basic_out = gr.JSON()
#         with gr.Tab("說明欄位（已去除基本欄位）"):
#             detail_out = gr.JSON()

#     btn.click(ui_extract, inputs=[doc_input, doc_type], outputs=[basic_out, detail_out])

# demo.launch()

#FastAPI：兩個 endpoint

app = FastAPI(title="RAG 公文擷取 API")

class BasicRequest(BaseModel):
    text: str

class DetailRequest(BaseModel):
    text: str
    doc_type: str
    k: int = 8

@app.post("/extract_basic")
def extract_basic(req: BasicRequest):
    return extract_basic_info_with_llm(req.text)

@app.post("/extract_detail")
def extract_detail(req: DetailRequest):
    return extract_detail_only(req.text, req.doc_type, k=req.k)

# （可選）如果你也想要一個一次回兩段的
@app.post("/extract_all")
def extract_all(req: DetailRequest):
    basic = extract_basic_info_with_llm(req.text)
    detail = extract_detail_only(req.text, req.doc_type, k=req.k)
    return {"基本欄位": basic, "說明欄位": detail}

#打開 http://localhost:8000/docs