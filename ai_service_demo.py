"""
AI 服务集合示例代码
适用于 Windows 32G 无GPU 环境
"""

import os
from typing import List, Optional

# ============ 1. 聊天模型（Ollama） ============
class ChatService:
    def __init__(self, model_name: str = "qwen3:4b"):
        import ollama
        self.client = ollama
        self.model = model_name
    
    def chat(self, message: str, history: Optional[List] = None) -> str:
        """聊天对话"""
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        response = self.client.chat(model=self.model, messages=messages)
        return response['message']['content']


# ============ 2. 嵌入模型（Embedding） ============
class EmbeddingService:
    def __init__(self, method: str = "ollama"):
        """
        method: "ollama" 或 "sentence-transformers"
        """
        self.method = method
        
        if method == "ollama":
            import ollama
            self.client = ollama
            self.model = "nomic-embed-text"  # 需要先 ollama pull
        else:
            from sentence_transformers import SentenceTransformer
            # 中文推荐 bge-small-zh-v1.5，英文推荐 all-MiniLM-L6-v2
            self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        if self.method == "ollama":
            embeddings = []
            for text in texts:
                response = self.client.embeddings(model=self.model, prompt=text)
                embeddings.append(response['embedding'])
            return embeddings
        else:
            return self.model.encode(texts).tolist()


# ============ 3. Rerank 模型 ============
class RerankService:
    def __init__(self, method: str = "local"):
        """
        method: "local" 或 "cohere"
        """
        self.method = method
        
        if method == "local":
            from sentence_transformers import CrossEncoder
            # self.model = CrossEncoder('BAAI/bge-reranker-base')
            self.model = CrossEncoder('models/bge-reranker-base')
        else:
            import cohere
            # 需要设置环境变量 COHERE_API_KEY
            self.client = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[dict]:
        """对文档进行重排序"""
        if self.method == "local":
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)
            
            results = [
                {"index": i, "text": doc, "score": float(score)}
                for i, (doc, score) in enumerate(zip(documents, scores))
            ]
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        else:
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-multilingual-v2.0"
            )
            return [
                {"index": r.index, "text": documents[r.index], "score": r.relevance_score}
                for r in response.results
            ]


# ============ 4. 语音转文字（ASR） ============
class Speech2TextService:
    def __init__(self, model_size: str = "base"):
        """
        model_size: tiny, base, small, medium
        CPU 推荐 base 或 small
        """
        from faster_whisper import WhisperModel
        # 使用 CPU，int8 量化
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def transcribe(self, audio_path: str, language: str = "zh") -> str:
        """转录音频文件"""
        segments, info = self.model.transcribe(audio_path, language=language)
        text = " ".join([segment.text for segment in segments])
        return text.strip()


# ============ 5. 文字转语音（TTS） ============
class Text2SpeechService:
    def __init__(self, method: str = "pyttsx3"):
        """
        method: "pyttsx3"（离线快速） 或 "edge"（在线高质量）
        """
        self.method = method
        
        if method == "pyttsx3":
            import pyttsx3
            self.engine = pyttsx3.init()
            # 设置中文语音（如果有）
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
    
    def speak(self, text: str, output_file: Optional[str] = None):
        """文字转语音"""
        if self.method == "pyttsx3":
            if output_file:
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
            else:
                self.engine.say(text)
                self.engine.runAndWait()
        else:
            # edge-tts 异步方法
            import asyncio
            import edge_tts
            
            async def _speak():
                communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
                if output_file:
                    await communicate.save(output_file)
                else:
                    await communicate.save("temp_output.mp3")
                    # 播放音频
                    os.system(f"start temp_output.mp3")
            
            asyncio.run(_speak())


# ============ 6. 图片转文字（OCR） ============
class Image2TextService:
    def __init__(self, method: str = "tesseract"):
        """
        method: "tesseract"（OCR） 或 "ollama"（多模态模型）
        """
        self.method = method
        
        if method == "tesseract":
            import pytesseract
            # Windows 需要指定 tesseract 路径
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        else:
            import ollama
            self.client = ollama
            self.model = "llava:7b"  # 需要先 ollama pull llava:7b
    
    def extract_text(self, image_path: str) -> str:
        """从图片提取文字"""
        if self.method == "tesseract":
            import pytesseract
            from PIL import Image
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            return text.strip()
        else:
            # 使用 Ollama 多模态模型
            with open(image_path, 'rb') as f:
                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': '请描述这张图片的内容',
                        'images': [image_path]
                    }]
                )
            return response['message']['content']


# ============ 使用示例 ============
if __name__ == "__main__":
    # 1. 聊天
    print("=== 聊天模型测试 ===")
    chat = ChatService()
    response = chat.chat("你好，请介绍一下自己")
    print(f"回复: {response}\n")
    
    # 2. 嵌入
    print("=== 嵌入模型测试 ===")
    # 使用 sentence-transformers（不需要额外下载 ollama 模型）
    # embedding = EmbeddingService(method="sentence-transformers")
    embedding = EmbeddingService()
    vectors = embedding.embed(["你好世界", "Hello World"])
    print(f"向量维度: {len(vectors[0])}\n")
    
    # 3. Rerank
    print("=== Rerank 模型测试 ===")
    rerank = RerankService(method="local")
    docs = [
        "Python 是一种编程语言",
        "今天天气很好",
        "机器学习是人工智能的一个分支"
    ]
    results = rerank.rerank("什么是编程", docs, top_k=2)
    print(f"最相关文档: {results[0]['text']}\n")
    
    # 4. TTS
    print("=== TTS 测试 ===")
    tts = Text2SpeechService(method="pyttsx3")
    tts.speak("你好，这是一个测试", output_file="test_output.mp3")
    print("语音已保存到 test_output.mp3\n")
    
    # 注意：ASR 和 Image2Text 需要实际的音频/图片文件才能测试
    print("其他服务需要实际文件进行测试")
