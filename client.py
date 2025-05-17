import configparser
import requests
import logging
import httpx
import asyncio

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/client.log"),
        logging.StreamHandler()
    ]
)
config = configparser.ConfigParser()
config.read('config.ini')

# 自动加载所有模型配置
LLM_BACKENDS = {}  # model_name => {api_url, api_key(optional)}

for section in config.sections():
    if section.startswith("llm."):
        model_id = section.split("llm.")[1]
        LLM_BACKENDS[model_id] = {
            "api_url": config.get(section, "api_url"),
            "api_key": config.get(section, "api_key", fallback=None)
        }


def get_model_list():
    models = []
    for backend, info in LLM_BACKENDS.items():
        try:
            url = info["api_url"].replace("/chat/completions", "/models")
            headers = {}
            if info.get("api_key"):
                headers["Authorization"] = f"Bearer {info['api_key']}"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("data", []):
                models.append({
                    "id": f"{backend}:{m['id']}",
                    "object": "model",
                    "created": m.get("created", 1680000000),
                    "owned_by": backend
                })
        except Exception as e:
            logging.error(f"[{backend}] 获取模型列表失败: {e}")
    return models


async def stream_chat_completion(messages, model: str, temperature=0.7):
    if ':' not in model:
        raise ValueError("模型名必须包含 backend 前缀，如 'openai:gpt-4'")

    backend, real_model = model.split(":", 1)
    if backend not in LLM_BACKENDS:
        raise ValueError(f"未知模型后端 '{backend}'，请检查配置")

    backend_info = LLM_BACKENDS[backend]
    url = backend_info["api_url"]
    headers = {"Accept": "text/event-stream"}
    if backend_info.get("api_key"):
        headers["Authorization"] = f"Bearer {backend_info['api_key']}"
    
    payload = {
        "model": real_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n"
                    

def send_chat_completion(messages, model: str, temperature=0.7):
    try:
        if ':' not in model:
            raise ValueError("模型名必须包含 backend 前缀，如 'openai:gpt-4'")

        backend, real_model = model.split(":", 1)
        if backend not in LLM_BACKENDS:
            raise ValueError(f"未知模型后端 '{backend}'，请检查配置")

        backend_info = LLM_BACKENDS[backend]

        payload = {
            "model": real_model,
            "messages": messages,
            "temperature": temperature
        }

        headers = {}
        if backend_info.get("api_key"):
            headers["Authorization"] = f"Bearer {backend_info['api_key']}"

        resp = requests.post(backend_info["api_url"], json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"调用模型 {model} 失败: {e}")
        return {"error": str(e)}
    