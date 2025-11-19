# 文档向量检索系统

一个基于ChromaDB和Sentence Transformers的简单文档向量检索系统，支持PDF、DOCX和TXT文档的上传、向量化和检索。

## 功能特性

1. 支持多种文档格式（PDF、DOCX、TXT）
2. 自动文档分块处理
3. 使用Sentence Transformers生成文档嵌入向量
4. 使用ChromaDB存储和检索向量
5. 提供RESTful API接口
6. 提供简单的Web界面用于测试

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
cd api
python document_api.py
```

服务将在 `http://localhost:8001` 启动。

## 使用方法

### 1. Web界面

访问 `http://localhost:8001/static/index.html` 使用Web界面上传文档和检索内容。

### 2. API接口

#### 上传文档

```bash
curl -X POST "http://localhost:8001/upload/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

#### 搜索文档

```bash
curl -X POST "http://localhost:8001/search/" \
     -H "Content-Type: application/json" \
     -d '{"query": "你的搜索关键词", "n_results": 5}'
```

### 3. API文档

访问 `http://localhost:8001/docs` 查看自动生成的API文档。

## 项目结构

```
api/
├── document_api.py         # FastAPI服务主文件
├── document_processor.py    # 文档处理逻辑
├── static/                 # 静态文件目录
│   └── index.html          # Web界面
└── chroma_db/              # ChromaDB数据存储目录（运行时自动生成）
```

## 技术栈

- FastAPI: Web框架
- ChromaDB: 向量数据库
- Sentence Transformers: 嵌入模型
- Langchain: 文本分割工具
- PyPDF2, python-docx: 文档解析库

## 注意事项

1. 文档上传后会自动分块并生成向量存储到ChromaDB中
2. 搜索时会根据语义相似度返回最相关的结果
3. 向量数据库数据会持久化存储在 `chroma_db` 目录中


python -m venv ck-ai
.\ck-ai\Scripts\activate
pip install -r requirements.txt
如果pip无法识别，则重新安装 python -m ensurepip --upgrade

再更新pip python -m pip install --upgrade pip

pip list > 查看安装包情况
qwen3:4b 模型 通过ollama 安装
ollama pull nomic-embed-text > 通过ollama安装向量模型
bge-reranker-base 到huggingface下载的模型到本地 （rerank 模型）
faster-whisper-base 到huggingface下载的模型到本地 （ASR 语音转文字）
Tesseract-OCR 本地安装该模型本地引用：https://github.com/UB-Mannheim/tesseract/wiki/Installation