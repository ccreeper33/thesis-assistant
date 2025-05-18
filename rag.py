from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import json
import configparser
import logging
import faiss

# 配置日志系统
logger = logging.getLogger(__name__)

# 加载配置文件
config = configparser.ConfigParser()
config.read('config.ini')

vector_path = config.get('rag', 'vector_store_path')
docs_path = config.get('rag', 'docs_path')
embedding_model = config.get('rag', 'embedding_model')
_vector_store_instance = None

# 加载嵌入模型
logger.info("正在加载嵌入模型")
try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
except Exception as e:
    logger.error(f"加载嵌入模型失败: {e}")


def load_metadata(pdf_path):
    """
    根据 PDF 文件路径加载同名 JSON 元数据文件
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(os.path.dirname(pdf_path), f"{base_name}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载元数据失败: {e}")
    return None


def build_vector_store():
    """
    生成 RAG 文档向量库
    """
    try:
        docs = []

        # 递归遍历所有目录，查找 .pdf 文件
        for root, dirs, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    logger.info(f"正在处理文件: {pdf_path}")

                    # 加载对应 metadata
                    metadata = load_metadata(pdf_path)

                    # 加载 PDF 内容
                    try:
                        loader = PyPDFLoader(pdf_path)
                        raw_docs = loader.load()
                    except Exception as e:
                        logger.warning(f"加载 PDF 失败: {pdf_path}, 原因: {e}")
                        continue

                    # 添加 metadata 或标记为 guidebook
                    for doc in raw_docs:
                        if metadata:
                            try:
                                doc.metadata.update({
                                    "title": metadata["title"],
                                    "id": metadata["id"],
                                    "authors": ", ".join(metadata["authors"]),
                                    "published": metadata["published"],
                                    "source_type": "arxiv"
                                })
                            except KeyError as ke:
                                logger.error(
                                    f"metadata 缺少必要字段: {ke}，文件: {pdf_path}")
                        else:
                            doc.metadata["source_type"] = "guidebook"

                    docs.extend(raw_docs)

        if not docs:
            logger.error("未加载任何文档，无法构建向量库")
            return

        logger.info(f"共加载了 {len(docs)} 个文档")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(config.get('rag', 'chunk_size', fallback='500')),
            chunk_overlap=int(config.get(
                'rag', 'chunk_overlap', fallback='100'))
        )
        chunks = splitter.split_documents(docs)

        logger.info(f"准备构建向量库， chunk 数: {len(chunks)}")

        sample_embedding = embeddings.embed_query("test")
        dimension = len(sample_embedding)
        db = FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(dimension),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        logger.info("已创建空白 FAISS 向量库")

        # ✅ 逐条插入 document，跳过无效内容
        success_count = 0
        fail_count = 0

        for i, doc in enumerate(chunks):
            content = doc.page_content
            metadata = doc.metadata

            if not content.strip():
                logger.warning(f"Chunk {i+1} 内容为空，跳过插入")
                fail_count += 1
                continue

            try:
                db.add_documents([doc])
                success_count += 1
            except Exception as e:
                logger.error(f"插入 chunk 失败（编号 {i+1} ）:{e}")
                fail_count += 1

        logger.info(f"成功插入 {success_count} 个 chunk ，失败 {fail_count} 个")

        # ✅ 保存向量库
        db.save_local(vector_path)
        logger.info("向量库构建完成并已保存")

    except Exception as e:
        logger.exception(f"构建向量库时发生未知错误: {e}")


def load_vector_store():
    """
    加载向量库
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        try:
            _vector_store_instance = FAISS.load_local(vector_path, embeddings,
                                                      allow_dangerous_deserialization=True)
            logger.info("向量库加载成功。")
        except Exception as e:
            logger.error(f"加载向量库失败: {e}")
            _vector_store_instance = None
    return _vector_store_instance


async def retrieve_context(query, k=5):
    """
    从向量库中搜索最接近 query 的 k 个上下文片段
    """
    vector_store = load_vector_store()
    if not vector_store:
        return []

    results = vector_store.similarity_search(query, k=k)
    contexts = []

    for r in results:
        contexts.append({
            "content": r.page_content,
            "metadata": r.metadata
        })

    logger.info(f"检索到 {len(contexts)} 个上下文片段用于查询: {query}")
    return contexts


async def build_prompt(user_query):
    """
    将用户提示词与 RAG 检索到的上下文拼接
    """
    logger.info(f"开始拼接用户请求：{user_query}")
    results = await retrieve_context(user_query)
    context_with_citations = []

    for i, item in enumerate(results):
        content = item["content"]
        meta = item["metadata"]

        if meta.get("source_type") == "arxiv":
            title = meta.get("title")
            paper_id = meta.get("id")
            authors = meta.get("authors")
            published = meta.get("published")

            ref = f"[{i+1}]: {title}, {authors}, arXiv:{paper_id}, {published}"
            block = f"[{i+1}] {content}\n[Reference {i+1}]: {ref}"
        else:
            block = content

        context_with_citations.append(block)

    full_prompt = "\n\n".join(context_with_citations)
    full_prompt += f"\n\n[Question]\n{user_query}\n\n[Instructions]\n"
    full_prompt += (
        "Answer the question based on the provided context below.  "
        "When using information from the context, mark the corresponding citation number in order of appearance (e.g., [1], [2], [3]).  "
        "At the end of your answer, include a \"References\" section that lists the full reference details in the order they appeared.  "
        "Do not include any information that is not supported by the context."
    )

    return full_prompt
