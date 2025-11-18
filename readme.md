
### python -m venv ck-ai
### .\ck-ai\Scripts\activate
### pip install -r requirements.txt
> 如果pip无法识别，则重新安装 python -m ensurepip --upgrade

> 再更新pip python -m pip install --upgrade pip

###  pip list > 查看安装包情况
### qwen3:4b 模型 通过ollama 安装
### ollama pull nomic-embed-text > 通过ollama安装向量模型 
### bge-reranker-base 到`huggingface`下载的模型到本地 （rerank 模型）
### faster-whisper-base 到`huggingface`下载的模型到本地 （ASR 语音转文字）
### Tesseract-OCR 本地安装该模型本地引用：https://github.com/UB-Mannheim/tesseract/wiki/Installation