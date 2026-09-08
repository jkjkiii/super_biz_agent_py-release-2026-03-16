"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops
from app.core.milvus_client import milvus_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"{config.app_name} v{config.app_version} 启动中...")
    logger.info(f"环境: {'开发' if config.debug else '生产'}")
    logger.info(f"监听地址: http://{config.host}:{config.port}")
    logger.info(f"API 文档: http://{config.host}:{config.port}/docs")

    # 连接 Milvus（后台重试，不阻塞启动）
    logger.info("正在连接 Milvus...")
    milvus_connected = False
    max_retries = 3
    retry_delay = 2  # 秒

    for attempt in range(1, max_retries + 1):
        try:
            milvus_manager.connect()
            milvus_connected = True
            logger.info("Milvus 连接成功")
            break
        except Exception as e:
            logger.warning(
                f"Milvus 连接失败 (第 {attempt}/{max_retries} 次): {e}"
            )
            if attempt < max_retries:
                logger.info(f"   {retry_delay} 秒后重试...")
                await asyncio.sleep(retry_delay)
            else:
                logger.warning(
                    f"Milvus 连接失败，已重试 {max_retries} 次。"
                    f"服务将继续启动，依赖 Milvus 的功能将不可用。"
                )

    if not milvus_connected:
        logger.warning("注意：Milvus 未连接，RAG 检索功能将不可用")

    logger.info("=" * 60)

    yield

    # 关闭时执行
    if milvus_connected:
        logger.info("正在关闭 Milvus 连接...")
        try:
            milvus_manager.close()
            logger.info(f"{config.app_name} 关闭")
        except Exception as e:
            logger.error(f"关闭 Milvus 连接时出错: {e}")
    else:
        logger.info(f"{config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
