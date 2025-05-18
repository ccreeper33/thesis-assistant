import uvicorn
import argparse
import os
import configparser
import logging
from datetime import datetime


# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 配置日志系统
log_dir = config.get('logging', 'log_dir', fallback='logs')
log_level_str = config.get('logging', 'log_level', fallback='INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)


def start_server():
    """
    启动 FastAPI 服务
    """
    host = config.get("api", "host")
    port = config.getint("api", "port")
    logging.info(f"正在启动服务器: {host}:{port}")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_config=None
    )


def rebuild_vector_store():
    """
    清除已有向量库并重新构建
    """
    vector_path = config.get('rag', 'vector_store_path')

    index_file = os.path.join(vector_path, "index.faiss")
    pkl_file = os.path.join(vector_path, "index.pkl")

    if os.path.exists(index_file):
        os.remove(index_file)
        logging.info(f"已删除旧的 FAISS 索引文件: {index_file}")
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
        logging.info(f"已删除旧的元数据文件: {pkl_file}")

    from rag import build_vector_store
    build_vector_store()


def clean_cache():
    """
    清理缓存文件
    """
    cache_dir = "__pycache__"
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
        os.rmdir(cache_dir)
        logging.info(f"已删除缓存目录: {cache_dir}")

    log_dir = "logs"
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))
        os.rmdir(log_dir)
        logging.info(f"已删除日志目录: {log_dir}")


def show_help():
    print("""
可用命令:

  serve           启动 FastAPI 服务器
  build           构建 RAG 向量库（保留旧库）
  rebuild         重建 RAG 向量库（清除旧库）
  help            显示帮助信息
    """)


def main():
    parser = argparse.ArgumentParser(description="学术写作辅助系统 CLI")
    parser.add_argument("command", nargs="?", default="help", help="要执行的命令")
    args = parser.parse_args()

    command = args.command.lower()

    if command == "serve":
        start_server()

    elif command == "build":
        from rag import build_vector_store
        build_vector_store()

    elif command == "rebuild":
        rebuild_vector_store()

    elif command == "help" or not command:
        show_help()

    else:
        logging.error(f"未知命令: {command}")
        show_help()


if __name__ == "__main__":
    main()
