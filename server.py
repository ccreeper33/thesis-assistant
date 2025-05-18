from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from rag import build_prompt
from client import send_chat_completion, get_model_list, stream_chat_completion
import configparser
import logging

# 配置日志系统
logger = logging.getLogger(__name__)

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 加载系统提示词路径、温度、监听地址
sys_prompt_path = config.get('settings', 'sys_prompt_path')
temperature = float(config.get('settings', 'temperature'))
host = config.get("api", "host")
port = int(config.get("api", "port"))

# 读取系统提示词
try:
    with open(sys_prompt_path, 'r', encoding='utf-8') as f:
        sys_prompt = f.read()
    logger.info(f"系统提示词加载成功: {sys_prompt_path}")
except FileNotFoundError:
    logger.error(f"系统提示文件未找到，请检查 {sys_prompt_path} 是否存在。")
    sys_prompt = ""

# 初始化 FastAPI 实例
app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """
    主对话接口，支持流式和非流式返回
    """
    try:
        data = await request.json()
        user_message = data["messages"][-1]["content"]
        model = data["model"]
        temperature = float(data.get("temperature", 0.7))
        stream = data.get("stream", False)

        logger.info(f"收到聊天请求: model={model}, stream={stream}")
        logger.debug(f"用户消息: {user_message}")

        # 构造提示词
        user_prompt = await build_prompt(user_message)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 流式响应模式
        if stream:
            async def event_generator():
                async for chunk in stream_chat_completion(messages, model, temperature):
                    yield chunk
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # 非流式响应模式
        else:
            return await send_chat_completion(messages, model, temperature)

    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        return {"error": str(e)}


@app.get("/v1/models")
async def list_models():
    """
    获取所有模型列表
    """
    logger.info("收到模型列表请求")
    return {
        "object": "list",
        "data": await get_model_list()
    }
