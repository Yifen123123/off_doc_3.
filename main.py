# main.py
"""
RAG 公文擷取 FastAPI 服務

啟動方式（確保已經先跑過 build_index.py 建好 rag_chroma）：

    uvicorn main:app --reload

"""

import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ===== 日誌設定 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 路徑與模型設定 =====
BASE_DIR = "rag_docs"
STRUCTURE_DIR = os.path.join(BASE_DIR, "structure")
CHROMA_DIR = "rag_chroma"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
OLLAMA_MODEL = "qwen2.5:14b-instruct"

# ====== 初始化 LLM / Embedding / 向量庫 ======
logger.info("初始化 LLM：%s", OLLAMA_MODEL)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

logger.info("初始化 Embedding 模型：%s", EMBEDDING_MODEL_NAME)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

logger.info("載入既有 Chroma 向量庫：%s", CHROMA_DIR)
vectordb = Chroma(
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR,
)

# ====== 公文類型 Enum 與對應 structure 檔 ======

class DocType(str, Enum):
    DETAIN = "扣押命令"
    QUERY = "保單查詢"
    NOTE = "保單註記"
    FREEZE = "保單凍結"
    UNFREEZE = "解除扣押"
    RECEIVE = "收取命令"
    PAY = "給付命令"
    TRANSFER = "保單移轉"
    OTHER = "其他"


DOC_TYPE_TO_STRUCTURE: Dict[str, str] = {
    "扣押命令": "扣押命令_structure.txt",
    "保單查詢": "保單查詢_structure.txt",
    "保單註記": "保單註記_structure.txt",
    "保單凍結": "保單凍結_structure.txt",
    "解除扣押": "解除扣押_structure.txt",
    "收取命令": "收取命令_structure.txt",
    "給付命令": "給付命令_structure.txt",
    "保單移轉": "保單移轉_structure.txt",
    "其他": "其他_structure.txt",
}

# ====== RAG 檢索 ======

def retrieve_context(query: str, k: int = 6, category: Optional[str] = None) -> str:
    """從 Chroma 擷取前 k 筆相似內容，選擇性依 category 篩選。"""
    if category:
        docs = vectordb.similarity_search(
            query, k=k, filter={"category": category}
        )
    else:
        docs = vectordb.similarity_search(query, k=k)

    context_chunks = []
    for d in docs:
        prefix = f"[{d.metadata.get('category', '')}/{d.metadata.get('filename', '')}]"
        context_chunks.append(f"{prefix} {d.page_content}")
    return "\n\n".join(context_chunks)


def load_structure_text(doc_type: str) -> str:
    """依公文類型載入對應的 JSON schema/結構說明檔。"""
    fname = DOC_TYPE_TO_STRUCTURE.get(doc_type)
    if not fname:
        raise ValueError(f"未知的公文類型: {doc_type}")
    path = os.path.join(STRUCTURE_DIR, fname)
    if not os.path.isfile(path):
        raise ValueError(f"找不到對應的 structure 檔案: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ====== 基本欄位擷取（含 heuristic 覆寫） ======

SYSTEM_PROMPT_BASIC = """
你是一個台灣法院/機關公文的「基本欄位擷取助手」。

任務：只擷取以下四個欄位，並以 JSON 輸出：
- 基準日
- 來函機關
- 收文編號
- 查詢對象（姓名加身分證字號，建議以「姓名, 身分證」格式輸出）

注意：
1. 基準日與收文編號以公文前半部的「日期行 + 鄰近純數字行」為主，
   若無法明確判斷，兩者填入「無明確記載」。
2. 收文編號不得是電話、傳真、保單號碼、統編、帳號等。
3. 嚴格輸出 JSON，不要包含多餘說明文字或註解。
""".strip()

# 日期模式（可視需要再補充）
DATE_PATTERNS = [
    r"中華民國\s*\d+\s*年\s*\d+\s*月\s*\d+\s*日",
    r"\d{3}\.\d{2}\.\d{2}",            # 例如 114.12.05
    r"\d{4}-\d{2}-\d{2}",             # 例如 2025-01-01
]

BANNED_KEYWORDS = [
    "電話", "傳真", "保單", "帳號", "統一編號", "統編", "代號",
    "股", "股數", "匯款", "金額",
]


def _is_date_line(line: str) -> bool:
    return any(re.search(pat, line) for pat in DATE_PATTERNS)


def _is_pure_number_line(line: str) -> bool:
    # 去掉空白後是否全為數字
    s = line.replace(" ", "")
    return s.isdigit()


def _has_banned_keyword(line: str) -> bool:
    return any(kw in line for kw in BANNED_KEYWORDS)


def find_date_and_receipt_pair(raw_doc: str):
    """
    在公文前半部尋找：
    - 一行「看起來是日期」；
    - 其之後最近的一行「純數字且不含禁字」作為收文編號。
    找不到時回傳 (None, None)。
    """
    lines = [ln.strip() for ln in raw_doc.splitlines() if ln.strip()]
    if not lines:
        return None, None

    # 只看前半段
    half = max(1, len(lines) // 2)
    head_lines = lines[:half]

    for i, line in enumerate(head_lines):
        if _is_date_line(line):
            # 向下找最近的純數字行
            for j in range(i + 1, min(i + 6, len(head_lines))):
                candidate = head_lines[j]
                if _is_pure_number_line(candidate) and not _has_banned_keyword(candidate):
                    return line, candidate
    return None, None


def normalize_target_field(s: str) -> str:
    """
    簡單把「查詢對象」統一成「姓名, 身分證」的字串。
    若無法判斷就原樣回傳，避免過度破壞資訊。
    """
    s = s.strip()
    # 嘗試以逗號或頓號分隔
    for sep in [",", "，", "、"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                return f"{parts[0]}, {parts[1]}"
    return s


def extract_basic_info_with_llm(raw_doc: str) -> Dict[str, Any]:
    """呼叫 LLM 擷取基本欄位，並用 heuristic 覆寫基準日與收文編號。"""
    user_prompt = f"""
{SYSTEM_PROMPT_BASIC}

【公文原文】
{raw_doc}

請回傳 JSON，格式如下：
{{
  "基準日": "",
  "來函機關": "",
  "收文編號": "",
  "查詢對象": ""
}}
""".strip()

    resp = llm.invoke(user_prompt).content.strip()
    # 移除 ```json code fence
    cleaned = (
        resp.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("解析基本欄位 JSON 失敗：%s\n原始回應：%s", e, resp)
        raise

    # heuristic 覆寫日期及收文編號
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

# ====== 說明欄位（detail） RAG 擷取 ======

SYSTEM_PROMPT_DETAIL = """
你是一個專門處理台灣法院與行政機關公文的「欄位擷取助手」。

任務：
1. 讀懂公文原文。
2. 結合提供的法律條文、概念與案例說明（RAG context），判斷各欄位內容。
3. 必須嚴格依照給定的 JSON schema（欄位名稱與結構）輸出。

輸出要求：
- 僅輸出 JSON，不要多餘文字或解釋。
- 若某欄位在公文中沒有明確記載，填入「無明確記載」或空陣列/空字串，視該欄位型別而定。
""".strip()


def extract_with_rag(raw_doc: str, doc_type: str, k: int = 8) -> str:
    """
    依公文類型 doc_type 讀取對應 structure 說明，
    再結合向量庫擷取的相關內容，一起丟給 LLM 產出 JSON。
    """
    structure_text = load_structure_text(doc_type)
    rag_context = retrieve_context(query=raw_doc, k=k, category=None)

    user_prompt = f"""
{SYSTEM_PROMPT_DETAIL}

公文類型：{doc_type}

【公文原文】
{raw_doc}

【相關知識（法律、概念、案例等，僅供你推理用）】
{rag_context}

【輸出格式與欄位說明（JSON schema）】
{structure_text}
""".strip()

    resp = llm.invoke(user_prompt).content
    return resp


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    嘗試從 LLM 回傳的文字中解析 JSON。
    - 先移除 ```json code fence
    - 若失敗就丟出例外，由上層轉成 HTTP 錯誤。
    """
    cleaned = (
        text.replace("```json", "")
        .replace("```", "")
        .strip()
    )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("解析 detail JSON 失敗：%s\n原始回應：%s", e, text)
        raise


BASIC_LIKE_KEYS = {"基準日", "來函機關", "收文編號", "查詢對象"}


def prune_basic_from_detail(obj: Any) -> Any:
    """
    從 detail JSON 中遞迴刪除與基本欄位重複的 key。
    保留其餘說明欄位即可。
    """
    if isinstance(obj, dict):
        return {
            k: prune_basic_from_detail(v)
            for k, v in obj.items()
            if k not in BASIC_LIKE_KEYS
        }
    elif isinstance(obj, list):
        return [prune_basic_from_detail(x) for x in obj]
    else:
        return obj


def extract_detail_only(raw_doc: str, doc_type: str, k: int = 8) -> Dict[str, Any]:
    """只擷取說明欄位：RAG + LLM + prune basic keys。"""
    detail_text = extract_with_rag(raw_doc, doc_type, k=k)
    detail_json = safe_parse_json(detail_text)
    detail_clean = prune_basic_from_detail(detail_json)
    return detail_clean

# ====== FastAPI schema 定義 ======

class BasicRequest(BaseModel):
    text: str = Field(
        ...,
        description="完整公文文字（含抬頭、主旨、說明等）",
        example="臺灣高雄地方法院民事執行處 執行命令\n114.12.05\n11400132\n...",
    )


class BaseDocText(BaseModel):
    text: str = Field(
        ...,
        description="完整公文文字（含抬頭、主旨、說明等）",
    )
    k: int = Field(
        8,
        description="RAG 檢索的前 k 筆相似內容（說明欄位用）",
        ge=1,
        le=20,
    )


class BasicFields(BaseModel):
    基準日: str
    來函機關: str
    收文編號: str
    查詢對象: Any


class DetailFields(BaseModel):
    data: Dict[str, Any]


class ExtractAllResponse(BaseModel):
    基本欄位: BasicFields
    說明欄位: DetailFields

# ====== 建立 FastAPI app ======

app = FastAPI(
    title="RAG 公文擷取 API",
    description="提供公文基本欄位與九類說明欄位的擷取服務。",
    version="0.2.0",
)


# ====== Endpoints ======

@app.post(
    "/basic",
    response_model=BasicFields,
    summary="擷取公文基本欄位",
    description="輸入完整公文文字，回傳基準日、來函機關、收文編號、查詢對象四個欄位。",
)
def extract_basic(req: BasicRequest):
    try:
        result = extract_basic_info_with_llm(req.text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM 回傳格式錯誤，無法解析基本欄位 JSON。",
        )
    return result


@app.post(
    "/detail/{doc_type}",
    response_model=DetailFields,
    summary="擷取指定類型公文的說明欄位",
    description="依照公文類型（九類之一），使用 RAG + LLM 擷取說明欄位。",
)
def extract_detail(doc_type: DocType, req: BaseDocText):
    try:
        detail = extract_detail_only(
            raw_doc=req.text,
            doc_type=doc_type.value,
            k=req.k,
        )
    except ValueError as e:
        # 包含未知 doc_type 或找不到 structure 檔案
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM 回傳格式錯誤，無法解析說明欄位 JSON。",
        )
    return {"data": detail}


@app.post(
    "/all/{doc_type}",
    response_model=ExtractAllResponse,
    summary="同時擷取基本欄位與說明欄位",
    description="先擷取基本欄位，再依公文類型擷取說明欄位，一次回傳。",
)
def extract_all(doc_type: DocType, req: BaseDocText):
    try:
        basic = extract_basic_info_with_llm(req.text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM 回傳格式錯誤，無法解析基本欄位 JSON。",
        )

    try:
        detail = extract_detail_only(
            raw_doc=req.text,
            doc_type=doc_type.value,
            k=req.k,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM 回傳格式錯誤，無法解析說明欄位 JSON。",
        )

    return {
        "基本欄位": basic,
        "說明欄位": {"data": detail},
    }

#執行：uvicorn main:app --relaod