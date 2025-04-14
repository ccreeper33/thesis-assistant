from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import configparser
import logging


# 配置日志，将日志保存到文件
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rag.log"),
        logging.StreamHandler()
    ]
)

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 从配置文件中获取向量存储路径
vector_path = config.get('rag', 'vector_store_path')

def build_vector_store():
    try:
        docs = []
        for file in os.listdir(config.get('rag', 'docs_path')):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(config.get('paths', 'docs_path'), file))
                docs.extend(loader.load())
        logging.info("Loaded all PDF documents")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        logging.info("Split documents into text chunks")

        embeddings = HuggingFaceEmbeddings(model_name=config.get('llm', 'embedding_model'))
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(vector_path)
        logging.info("Vector store built and saved.")
    except Exception as e:
        logging.error(f"Error building vector store: {e}")

def load_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.get('llm', 'embedding_model'))
        db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
        logging.info("Vector store loaded.")
        return db
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        return None

def retrieve_context(query, k=3):
    db = load_vector_store()
    if db:
        results = db.similarity_search(query, k=k)
        context = "\n\n".join([r.page_content for r in results])
        logging.info(f"Retrieved context for query: {query}")
        return context
    return ""

def build_prompt(user_query):
    context = retrieve_context(user_query)
    prompt = f"""[Context]
{context}

[Question]
{user_query}

[Answer]
"""
    logging.info(f"Built prompt for user query: {user_query}")
    return prompt
