# thesis-assistant

## 项目概述

本项目是一个大语言模型（LLM）辅助论文写作工具。

## 运行步骤

### 创建并激活虚拟环境（可选，推荐）

```bash
conda create -n thesis_assistant python=3.11
conda activate thesis_assistant
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 构建向量存储

```bash
python -c "from rag import build_vector_store; build_vector_store()"
```

该命令会读取 `docs` 目录下的所有 `PDF` 文档，将其分割成文本块，生成嵌入向量，并保存到 `vector_store` 目录中。

### 配置并启动llm后端

1. 将 `config.ini` 中，`[llm]` 块的 `api_url` 和 `llm_model` 修改为实际的内容。
2. 启动llm后端

### 启动服务

```bash
python main.py
```

### 发送请求

使用 `curl` 发送POST请求或使用支持openai api的前端

`curl` 示例：

```bash
$ curl http://localhost:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "messages": [
      {"role": "user", "content": "I am writing a research paper on gesture recognition. Could you help me generate a detailed outline for the paper, including section headings, related articles and brief descriptions"}
    ]
  }'
```
