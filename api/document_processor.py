"""
文档处理器模块
用于处理各种类型的文档并将其转换为向量存储到ChromaDB中
"""

import uuid
import os
from typing import List, Dict, Any
import tempfile
from pathlib import Path

# 尝试使用更新的 SQLite 版本
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx


class DocumentProcessor:
    """文档处理器类"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化文档处理器
        
        Args:
            persist_directory: ChromaDB持久化存储目录
        """
        # 初始化ChromaDB客户端（使用持久化模式）
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        从PDF文件中提取文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        从DOCX文件中提取文本
        
        Args:
            file_path: DOCX文件路径
            
        Returns:
            提取的文本内容
        """
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        从TXT文件中提取文本
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            提取的文本内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_text(self, file_path: str) -> str:
        """
        根据文件扩展名自动选择合适的文本提取方法
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取的文本内容
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            # 默认按文本文件处理
            return self.extract_text_from_txt(file_path)
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成较小的块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        return self.text_splitter.split_text(text)
    
    def add_document(self, file_path: str, doc_id: str = None) -> Dict[str, Any]:
        """
        添加文档到向量数据库
        
        Args:
            file_path: 文档路径
            doc_id: 文档ID（可选）
            
        Returns:
            添加结果
        """
        # 生成文档ID
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        # 提取文本
        text = self.extract_text(file_path)
        
        # 分割文本
        chunks = self.split_text(text)
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # 准备元数据
        metadatas = [
            {
                "source": file_path,
                "chunk_index": i
            } 
            for i in range(len(chunks))
        ]
        
        # 准备IDs
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # 添加到向量数据库
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "doc_id": doc_id,
            "chunks_added": len(chunks),
            "source": file_path
        }
    
    def search_documents(self, query: str, n_results: int = 5, distance_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询语句
            n_results: 返回结果数量
            distance_threshold: 距离阈值，低于此值的结果将被过滤掉（越小越相似）
            
        Returns:
            搜索结果列表
        """
        # 检查查询是否为空
        if not query or not query.strip():
            return []
        
        # 检查集合是否为空
        if self.collection.count() == 0:
            return []
        
        # 生成查询嵌入向量
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 格式化结果并应用距离阈值过滤
        formatted_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i] if 'distances' in results else None
            # 如果距离为None或者小于阈值，则添加到结果中
            if distance is None or distance <= distance_threshold:
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": distance
                })
            
        return formatted_results

# 使用示例
if __name__ == "__main__":
    # 初始化文档处理器
    processor = DocumentProcessor()
    
    # 添加文档示例（假设你有一个test.pdf文件）
    # result = processor.add_document("test.pdf")
    # print(f"文档添加结果: {result}")
    
    # 搜索文档示例
    # search_results = processor.search_documents("人工智能的发展历史")
    # print(f"搜索结果: {search_results}")