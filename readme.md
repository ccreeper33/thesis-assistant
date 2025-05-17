# 📚 Thesis Assistant - LLM 辅助论文写作系统

> **中文文档** | [English Version](README_en.md)

本项目是一个基于大语言模型（LLM）与检索增强生成（RAG）技术的**学术论文辅助写作工具**。它支持从本地 arXiv 论文中提取信息，并结合 LLM 生成结构化内容、润色文本、引用标注等。

---

## 🧩 项目特点

- ✅ 支持多模型后端（OpenAI、DeepSeek、阿里云通义千问、本地 vLLM 等）
- ✅ 基于 FAISS 的 RAG 检索系统，支持 GPU 加速
- ✅ 自动绑定 arXiv 论文 metadata，便于引用标注
- ✅ 支持命令行操作：构建向量库、启动服务、清理缓存等
- ✅ 提供统一 API 接口，兼容 OpenAI 标准格式
- ✅ 可扩展性强，支持后续集成 BibTeX 导出、OCR 扫描件处理等功能

---

## 📁 目录结构

```
.
├── client.py               # LLM 调用客户端
├── config.example.ini      # 配置文件示例（LLM 地址、API Key、路径等）
├── docs/                   # 存放 PDF 文档和对应 JSON 元数据
├── logs/                   # 日志输出目录
├── main.py                 # CLI 入口脚本
├── prompts/                # 提示词模板目录
│   └── sys_prompt.txt      # 系统提示词
├── rag.py                  # RAG 核心模块（检索 + 向量化 + prompt 构建）
├── requirements.txt        # Python 依赖包列表
├── server.py               # FastAPI 接口服务
└── vector_store/           # FAISS 向量库存储目录
```

---

## ⚙️ 运行步骤

### 1. 创建并激活虚拟环境（推荐）

```bash
conda create -n thesis_assistant python=3.11
conda activate thesis_assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 初始化配置文件

本项目使用 `config.ini` 管理 LLM 接口、模型路径等参数。**出于安全考虑，需自行修改示例文件 `config.example.ini`。**

#### ✅ 创建自己的配置文件：

```bash
cp config.example.ini config.ini
```

#### ✅ 编辑配置文件：

```bash
vim config.ini
```

填写你的 LLM API 地址、密钥、路径等信息。

### 4. 准备文档数据

将文档数据放入 `docs/` 目录中。

对于论文，每个 PDF 应有对应的同名 `.json` 文件，包含如下元数据：

```json
{
  "id": "xxx",
  "title": "xxx for xxx in xxx",
  "authors": [
    "xxx",
    "xxx",
    "xxx"
  ],
  "published": "yyyy-mm-dd"
}
```

### 5. 构建向量库

```bash
python main.py build
```

> 如果需要重建向量库（清除旧数据）：
```bash
python main.py rebuild
```

### 6. 启动服务

```bash
python main.py serve
```

默认监听地址为 `0.0.0.0:12345`，可通过 `config.ini` 修改。

### 7. 发送请求

你可以使用 `curl` 或任意支持 OpenAI API 的客户端发送请求：

```bash
curl http://localhost:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "messages": [
      {"role": "user", "content": "I am writing a research paper on gesture recognition. Could you help me generate a detailed outline for the paper, including section headings, related articles and brief descriptions"}
    ]
  }'
```

---

## 🛠️ 功能命令说明（CLI）

| 命令       | 描述                         |
|------------|------------------------------|
| `serve`   | 启动 FastAPI 服务             |
| `build`    | 构建向量库（保留旧库）        |
| `rebuild`  | 清除已有向量库并重新构建     |
| `help`     | 显示帮助信息                 |

---

## 📦 示例输出

当用户提问时，系统会自动检索上下文并生成带引用的回答：

```txt
[Context]
[1] Contrast-Enhanced Spectral Mammography (CESM) is a dual-energy mammographic technique that improves lesion visibility through the administration of an iodinated contrast agent...
[Reference 1]: Lesion-Aware Generative..., Aurora Rofena et al., arXiv:2505.03018v1, 2025-05-05

[Question]
What are the key contributions of Seg-CycleGAN?

[Instructions]
Answer the question based on the provided context. If you use information from a specific passage, cite its number (e.g., [1]). Do not include any information not present in the context.
```

---

