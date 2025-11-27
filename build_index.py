# build_index.py
"""
用來建立 / 更新 Chroma 向量庫的腳本。
只在你變更 rag_docs 底下的內容時手動執行一次即可：

    python build_index.py
"""

import os
import logging

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ===== 基本設定 =====
BASE_DIR = "rag_docs"
# 依你的實際資料夾命名調整，如果真的叫 "cocepts" 就維持一致
SUBDIRS = ["cases", "cocepts", "laws", "structure"]

CHROMA_DIR = "rag_chroma"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_documents() -> list:
    """讀取 rag_docs 下各子資料夾的 .txt 檔並加上 metadata。"""
    documents = []

    for sub in SUBDIRS:
        folder = os.path.join(BASE_DIR, sub)
        if not os.path.isdir(folder):
            logger.warning("子資料夾不存在，略過: %s", folder)
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(folder, fname)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()

            for d in docs:
                d.metadata["category"] = sub
                d.metadata["filename"] = fname

            documents.extend(docs)
            logger.info("載入檔案: %s (%d docs)", path, len(docs))

    logger.info("總共載入 %d 個 document", len(documents))
    return documents


def build_and_persist_chroma():
    """切 chunk、建立 Chroma 向量庫並 persist 到磁碟。"""
    logger.info("初始化 embedding 模型：%s", EMBEDDING_MODEL_NAME)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    docs = load_all_documents()
    if not docs:
        logger.error("沒有載入任何文件，請確認 rag_docs 內容。")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(docs)
    logger.info("完成切 chunk，總 chunk 數：%d", len(split_docs))

    # 如果已存在舊的索引，可以選擇先刪掉
    if os.path.isdir(CHROMA_DIR):
        logger.info("偵測到既有 Chroma 目錄，將覆蓋原索引：%s", CHROMA_DIR)

    logger.info("開始建立 Chroma 向量庫並寫入：%s", CHROMA_DIR)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    logger.info("Chroma 向量庫建立完成。")


if __name__ == "__main__":
    build_and_persist_chroma()


#啟動：uvicorn main:app --reload