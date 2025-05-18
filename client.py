import configparser
import requests
import logging
import httpx

# 配置日志系统
logger = logging.getLogger(__name__)

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 自动加载所有模型配置：支持多个后端模型
# 每个模型段落格式如 [llm.openai]、[llm.deepseek]
LLM_BACKENDS = {}  # model_name => {api_url, api_key(optional)}

for section in config.sections():
    if section.startswith("llm."):
        model_id = section.split("llm.")[1]
        LLM_BACKENDS[model_id] = {
            "api_url": config.get(section, "api_url"),
            "api_key": config.get(section, "api_key", fallback=None)
        }
        logger.info(
            f"已加载模型后端: {model_id} => {LLM_BACKENDS[model_id]['api_url']}")


async def get_model_list():
    """
    获取所有后端模型列表
    向每个已配置的 LLM 后端发送 /v1/models 请求，统一拼接前缀后返回
    """
    models = []
    for backend, info in LLM_BACKENDS.items():
        try:
            url = info["api_url"].replace("/chat/completions", "/models")
            headers = {}
            if info.get("api_key"):
                headers["Authorization"] = f"Bearer {info['api_key']}"

            logger.info(f"[{backend}] 请求模型列表: {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            data = resp.json()
            for m in data.get("data", []):
                model_id = f"thesis_assistant:{backend}:{m['id']}"
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": m.get("created", 1680000000),
                    "owned_by": backend
                })
                logger.debug(f"已发现模型: {model_id}")
        except Exception as e:
            logger.error(f"[{backend}] 获取模型列表失败: {e}")
    return models


async def stream_chat_completion(messages, model: str, temperature=0.7):
    """
    使用流式方式向指定模型发送聊天请求，并逐行返回数据流
    适用于支持 stream: true 的 LLM 接口
    """
    if ':' not in model:
        raise ValueError("模型名必须包含 backend 前缀，如 'openai:gpt-4'")

    _, backend, real_model = model.split(":", 2)
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

    logger.info(f"[{model}] 发起流式请求: {url}")
    logger.debug(f"请求信息:{messages}")
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                if line.strip() == "data: [DONE]":
                    logger.info(f"[{model}] 接收完成")
                    yield line + "\n"
                    break

                yield line + "\n"


async def send_chat_completion(messages, model: str, temperature=0.7):
    """
    发送普通的非流式聊天请求，返回完整响应
    用于不支持 stream 的模型或需要一次性返回内容的请求场景
    """
    try:
        if ':' not in model:
            raise ValueError("模型名必须包含 backend 前缀，如 'openai:gpt-4'")

        _, backend, real_model = model.split(":", 2)
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

        logger.info(f"[{model}] 发起非流式请求: {backend_info['api_url']}")
        resp = requests.post(
            backend_info["api_url"], json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"调用模型 {model} 失败: {e}")
        return {"error": str(e)}
