# AI 集合部署指南（Windows 32G 无GPU）

## 系统要求
- Windows 10/11
- 32GB RAM
- 无需 GPU
- 已安装 Ollama + qwen2.5:4b

## 安装步骤

### 1. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 2. 安装额外工具

#### Tesseract OCR（图片转文字）
下载并安装：https://github.com/UB-Mannheim/tesseract/wiki
安装后添加到系统 PATH，或在代码中指定路径

#### FFmpeg（音频处理）
下载：https://www.gyan.dev/ffmpeg/builds/
解压后添加 bin 目录到系统 PATH

### 3. 验证 Ollama
```bash
ollama list
# 应该能看到 qwen2.5:4b
```

## 模型推荐配置

### 嵌入模型（Embedding）
**推荐方案 1：使用 Ollama**
```python
# Ollama 支持嵌入，使用已有的 qwen2.5:4b 或下载专门的嵌入模型
ollama pull nomic-embed-text  # 轻量级嵌入模型
```

**推荐方案 2：使用 sentence-transformers**
```python
# 中文推荐：bge-small-zh-v1.5（小巧，CPU 友好）
# 英文推荐：all-MiniLM-L6-v2（轻量）
```

### 聊天模型
- 已有：qwen2.5:4b（4B 参数，32G 内存完全够用）
- 可选：ollama pull llama3.2:3b（更小更快）

### Rerank 模型
- 在线：Cohere API（需要注册获取免费 key）
- 本地：bge-reranker-base（通过 sentence-transformers 加载）

### TTS（文字转语音）
- 离线：pyttsx3（快速，质量一般）
- 在线：edge-tts（免费，质量好，需要网络）

### ASR（语音转文字）
- faster-whisper + base/small 模型（CPU 可跑）
- 首次使用会自动下载模型

### 图片转文字
- OCR：pytesseract（需要安装 Tesseract）
- 多模态：可以考虑 Ollama 的视觉模型（如 llava）

## 内存占用估算

| 模型类型 | 推荐模型 | 内存占用 |
|---------|---------|---------|
| 聊天模型 | qwen2.5:4b | ~4-6GB |
| 嵌入模型 | bge-small-zh | ~400MB |
| Rerank | bge-reranker-base | ~600MB |
| ASR | whisper-base | ~1GB |
| TTS | pyttsx3 | ~50MB |
| OCR | tesseract | ~100MB |
| **总计** | | **~6-8GB** |

32G 内存完全够用，还有充足余量！

## 注意事项

1. **首次运行会下载模型**，需要良好的网络连接
2. **Ollama 模型存储位置**：`C:\Users\你的用户名\.ollama\models`
3. **HuggingFace 模型缓存**：`C:\Users\你的用户名\.cache\huggingface`
4. **CPU 推理速度**：4B 模型在 CPU 上响应时间约 2-5 秒
5. **建议使用 SSD**：模型加载更快

## 可选优化

### 使用国内镜像加速
```bash
# HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com

# pip 镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Ollama 额外推荐模型
```bash
# 轻量级嵌入模型
ollama pull nomic-embed-text

# 视觉理解（图片转文字）
ollama pull llava:7b

# 更小的聊天模型
ollama pull qwen2.5:1.5b
```
