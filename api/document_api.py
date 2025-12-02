"""
文档处理API服务
提供文档上传和检索功能
"""

# 尝试使用更新的 SQLite 版本
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import os
from typing import List, Optional

from document_processor import DocumentProcessor

app = FastAPI(
    title="文档向量检索系统",
    description="一个简单的文档向量检索系统，支持PDF、DOCX和TXT文档",
    version="1.0.0"
)

# 挂载静态文件目录
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化文档处理器
doc_processor = DocumentProcessor("./chroma_db")


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str
    n_results: Optional[int] = 5
    distance_threshold: Optional[float] = 0.6


class SearchResult(BaseModel):
    """搜索结果模型"""
    id: str
    document: str
    metadata: dict
    distance: Optional[float]


class DocumentAddResponse(BaseModel):
    """文档添加响应模型"""
    doc_id: str
    chunks_added: int
    source: str


@app.get("/")
async def read_root():
    """根路径重定向到静态页面"""
    return {"message": "文档向量检索系统API服务已启动", 
            "docs": "/docs", 
            "static_page": "/static/index.html"}


@app.post("/upload/", response_model=DocumentAddResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并将其向量化存储
    
    Args:
        file: 上传的文档文件
        
    Returns:
        文档添加结果
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        # 处理文档
        result = doc_processor.add_document(tmp_file_path)
        
        # 删除临时文件
        os.unlink(tmp_file_path)
        
        return DocumentAddResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """
    搜索相关文档片段
    
    Args:
        request: 搜索请求参数
        
    Returns:
        搜索结果列表
    """
    try:
        # 检查查询是否为空
        if not request.query or not request.query.strip():
            return []
        
        results = doc_processor.search_documents(
            request.query, 
            request.n_results,
            request.distance_threshold
        )
        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    Returns:
        健康状态
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)