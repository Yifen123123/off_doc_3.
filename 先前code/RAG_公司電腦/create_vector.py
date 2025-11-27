import os
import sys
import shutil
from typing import List, Tuple

# --- 讓 requests/hf_hub 使用系統 CA 憑證（避免 SSL 驗證問題）---
import certifi
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
# 啟用更快的傳輸（若已安裝 hf_transfer）
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# 若你的公司已在系統層設定 HTTP(S)_PROXY，huggingface_hub 會自動信任它

from huggingface_hub import snapshot_download, hf_hub_download
from sentence_transformers import SentenceTransformer

# LangChain
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
try:
    # 建議：新套件（若未安裝會進到 except）
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # 相容舊版
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS


# =====================
# 設定
# =====================
UPLOAD_DIR = "uploaded_docs"
DB_DIR = "faiss_db"
DB_ZIP = "faiss_db.zip"

# 你要的模型（可改成 base 或 MiniLM 當備援）
PRIMARY_MODEL_ID = "intfloat/multilingual-e5-large"
BACKUP_MODEL_IDS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# 建資料夾
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"請將 .txt / .pdf / .docx 放到：{UPLOAD_DIR}")


# =====================
# 工具：下載 + 決定模型路徑
# =====================
def try_snapshot_download(model_id: str, local_dir: str) -> str:
    """
    用 snapshot_download 把整個模型抓到 local_dir。
    成功回傳實際存放路徑；失敗會丟例外。
    """
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return path


def gentle_probe_with_hf_hub_download(model_id: str) -> None:
    """
    透過 hf_hub_download 下載一個小檔案（例如 config.json）來測試 proxy/SSL，
    指定 trust_env=True 表示信任系統 proxy 設定。
    成功則代表連線 OK；失敗會丟例外。
    """
    # 不同模型檔名略有差異，優先嘗試常見 config.json
    candidates = ["config.json", "model.safetensors", "modules.json"]
    last_err = None
    for filename in candidates:
        try:
            _ = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                trust_env=True,     # 這行是你指定的重點
                resume_download=True,
            )
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"hf_hub_download 連線測試失敗：{last_err}")


def resolve_model_local_path(
    primary: str,
    backups: List[str],
    prefer_local_env: bool = True
) -> Tuple[str, bool]:
    """
    取得可用的本地模型資料夾，並判斷是否為 e5 系列（決定是否加 query/passage 前綴）。

    回傳：(local_path, is_e5)
    """
    # 1) 若有設定 EMBED_MODEL_DIR，優先使用（完全離線）
    local_env = os.environ.get("EMBED_MODEL_DIR")
    if prefer_local_env and local_env and os.path.isdir(local_env):
        is_e5 = ("e5" in os.path.basename(local_env).lower()) or ("intfloat" in local_env.lower())
        print(f"使用本地 EMBED_MODEL_DIR：{local_env}")
        return local_env, is_e5

    # 2) 依序嘗試 primary -> backups，下載到 models/<name>
    candidates = [primary] + list(backups)
    for mid in candidates:
        pretty_name = mid.replace("/", "-")
        local_dir = os.path.join("models", pretty_name)
        os.makedirs("models", exist_ok=True)

        # 已有快取就直接用
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            is_e5 = ("e5" in mid.lower()) or ("intfloat" in mid.lower())
            print(f"偵測到已存在的本地模型：{local_dir}")
            return local_dir, is_e5

        # 沒有就試著下載
        try:
            print(f"→ 嘗試 snapshot_download：{mid}")
            path = try_snapshot_download(mid, local_dir)
            is_e5 = ("e5" in mid.lower()) or ("intfloat" in mid.lower())
            print(f"✅ 下載完成：{mid} -> {path}")
            return path, is_e5
        except Exception as e1:
            print(f"⚠️ snapshot_download 失敗，改用 hf_hub_download 探測 proxy：{e1}")
            try:
                # 這一步只為了走一次代理，把通道打通；不會拿到整包模型
                gentle_probe_with_hf_hub_download(mid)
                # 探測成功後，再回頭用 snapshot_download 抓整包
                path = try_snapshot_download(mid, local_dir)
                is_e5 = ("e5" in mid.lower()) or ("intfloat" in mid.lower())
                print(f"✅ 下載完成（經 hf_hub_download 探測）：{mid} -> {path}")
                return path, is_e5
            except Exception as e2:
                print(f"❌ 下載 {mid} 仍失敗，嘗試下一個備援：{e2}")

    raise RuntimeError(
        "無法下載任何候選模型。\n"
        "請確認：\n"
        "1) 代理與憑證已正確設定（系統環境變數 HTTP_PROXY/HTTPS_PROXY）。\n"
        "2) 或改用離線：先把模型資料夾放到本機，設定 EMBED_MODEL_DIR 指向它。"
    )


# =====================
# 文本載入
# =====================
def load_txt_as_documents(path: str) -> List[Document]:
    encodings = ["utf-8", "utf-8-sig", "cp950"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": path, "encoding": enc})]
        except UnicodeDecodeError as e:
            last_err = e
    raise UnicodeDecodeError(f"無法用 {encodings} 任一編碼讀取：{path}\n最後錯誤：{last_err}")


def load_all_documents(folder_path: str) -> List[Document]:
    supported_ext = {".txt", ".pdf", ".docx"}
    documents: List[Document] = []
    files = sorted(os.listdir(folder_path))
    if not files:
        print(f"⚠️ 資料夾 {folder_path} 是空的。")
        return documents

    for file in files:
        if file.startswith("."):
            continue
        path = os.path.join(folder_path, file)
        _, ext = os.path.splitext(file.lower())
        if ext not in supported_ext:
            print(f"↪️ 略過不支援檔案：{file}")
            continue

        try:
            if ext == ".txt":
                docs = load_txt_as_documents(path)
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata = {**(d.metadata or {}), "source": path}
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata = {**(d.metadata or {}), "source": path}
            else:
                docs = []
            documents.extend(docs)
            print(f"✅ 載入成功：{file}（新增 {len(docs)} 筆）")
        except Exception as e:
            print(f"❌ 載入失敗：{file} -> {e}", file=sys.stderr)
    return documents


# =====================
# 嵌入器（用本地模型路徑）
# =====================
class SentenceTransformerEmbeddings:
    def __init__(self, local_model_dir: str, is_e5: bool):
        self.model = SentenceTransformer(local_model_dir)
        self.is_e5 = is_e5

    def embed_documents(self, texts: List[str]):
        if self.is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(
            texts, show_progress_bar=True, convert_to_tensor=False, normalize_embeddings=True
        )

    def embed_query(self, text: str):
        q = f"query: {text}" if self.is_e5 else text
        return self.model.encode(q, convert_to_tensor=False, normalize_embeddings=True)


# =====================
# 主流程
# =====================
def main():
    # 0) 顯示關鍵版本（debug用）
    try:
        import langchain, langchain_community
        print("[Versions]",
              "langchain=", getattr(langchain, "__version__", "unknown"),
              "langchain-community=", getattr(langchain_community, "__version__", "unknown"))
    except Exception:
        pass

    # 1) 載入文件
    docs = load_all_documents(UPLOAD_DIR)
    if not docs:
        print("⚠️ 沒有可用文件，流程結束。")
        return

    # 2) 分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"📄 已切出 {len(split_docs)} 個 chunks。")

    # 3) 取得模型（走 proxy / 離線皆可）
    try:
        local_model_dir, is_e5 = resolve_model_local_path(PRIMARY_MODEL_ID, BACKUP_MODEL_IDS)
    except Exception as e:
        print(f"❌ 模型下載/解析失敗：{e}", file=sys.stderr)
        return

    # 4) 建立向量庫
    print(f"🔎 使用模型：{local_model_dir}（is_e5={is_e5}）")
    embedding = SentenceTransformerEmbeddings(local_model_dir, is_e5)
    vectorstore = FAISS.from_documents(split_docs, embedding)
    print("✅ 向量索引完成。")

    # 5) 儲存
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    vectorstore.save_local(DB_DIR)
    print(f"💾 已儲存向量庫：{DB_DIR}")

    # 6) 壓縮備份
    if os.path.exists(DB_ZIP):
        os.remove(DB_ZIP)
    shutil.make_archive(DB_DIR, 'zip', DB_DIR)
    print(f"📦 已輸出：{DB_ZIP}")


if __name__ == "__main__":
    main()
