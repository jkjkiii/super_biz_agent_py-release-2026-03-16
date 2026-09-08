@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ====================================
echo 启动 SuperBizAgent 服务
echo ====================================
echo.

REM 总步骤数
set TOTAL_STEPS=8

REM ============================================
REM 步骤 1: 检查包管理器
REM ============================================
echo [1/%TOTAL_STEPS%] 检查包管理器...
where uv >nul 2>nul
if errorlevel 1 (
    echo [信息] uv 未安装，将使用传统 pip 方式
    set USE_UV=0
) else (
    echo [成功] 检测到 uv 包管理器
    set USE_UV=1
)
echo.

REM ============================================
REM 步骤 2: 查找合适的 Python 版本 (需要 >= 3.11)
REM ============================================
echo [2/%TOTAL_STEPS%] 查找 Python 3.11+ ...

set PYTHON_EXE=
set PYTHON_OK=0

REM 优先检查 python 命令版本
python --version >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>nul
    if not errorlevel 1 (
        set PYTHON_EXE=python
        set PYTHON_OK=1
        echo [成功] 使用系统默认 Python
    ) else (
        echo [信息] 系统 Python 版本过低，查找其他 Python 3.11+ ...
    )
)

REM 尝试 Anaconda super_biz_agent 环境
if "!PYTHON_OK!"=="0" (
    if exist "D:\ProgramData\Anaconda3\envs\super_biz_agent\python.exe" (
        D:\ProgramData\Anaconda3\envs\super_biz_agent\python.exe -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>nul
        if not errorlevel 1 (
            set PYTHON_EXE=D:\ProgramData\Anaconda3\envs\super_biz_agent\python.exe
            set PYTHON_OK=1
            echo [成功] 使用 Anaconda: super_biz_agent
        )
    )
)

REM 尝试 Python Launcher (py -3.11 / py -3.12 / py -3.13)
if "!PYTHON_OK!"=="0" (
    for %%v in (3.13 3.12 3.11) do (
        py -%%v --version >nul 2>nul
        if not errorlevel 1 (
            if "!PYTHON_OK!"=="0" (
                set PYTHON_EXE=py -%%v
                set PYTHON_OK=1
                echo [成功] 使用 Python Launcher: py -%%v
            )
        )
    )
)

if "!PYTHON_OK!"=="0" (
    echo [错误] 未找到 Python 3.11+
    echo.
    echo 请先安装 Python 3.11 或更高版本：
    echo   1. 官网下载: https://www.python.org/downloads/
    echo   2. 或 Anaconda: conda create -n super_biz_agent python=3.11
    echo.
    pause
    exit /b 1
)

echo.

REM ============================================
REM 步骤 3: 创建/同步虚拟环境
REM ============================================
echo [3/%TOTAL_STEPS%] 创建/同步虚拟环境...

REM 检查现有 venv 的 Python 版本
set VENV_OK=0
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>nul
    if not errorlevel 1 (
        set VENV_OK=1
    ) else (
        echo [信息] 现有虚拟环境版本过低，将重建...
        rmdir /s /q .venv
    )
)

if "!VENV_OK!"=="1" (
    echo [信息] 虚拟环境已存在，检查更新...
    if "%USE_UV%"=="1" (
        uv sync 2>nul
        if errorlevel 1 (
            echo [警告] uv sync 失败，使用 pip 更新...
            .venv\Scripts\python.exe -m pip install -e .
        ) else (
            echo [成功] 使用 uv 同步完成
        )
    ) else (
        echo [信息] pip 检查更新（可能需要几分钟）...
        .venv\Scripts\python.exe -m pip install -e .
    )
) else (
    echo [信息] 创建新的虚拟环境...
    !PYTHON_EXE! -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [信息] 安装项目依赖（首次可能需要几分钟，请耐心等待）...
    .venv\Scripts\python.exe -m pip install --upgrade pip -q
    .venv\Scripts\python.exe -m pip install -e .
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
    echo [成功] 虚拟环境创建完成
)

echo [成功] 虚拟环境就绪
echo.

REM 设置 Python 命令
set PYTHON_CMD=.venv\Scripts\python.exe

REM ============================================
REM 步骤 4: 启动 Milvus 向量数据库
REM ============================================
echo [4/%TOTAL_STEPS%] 启动 Milvus 向量数据库...

REM 检查 Docker 是否运行
docker info >nul 2>nul
if errorlevel 1 (
    echo [错误] Docker Desktop 未运行！
    echo [提示] 请先启动 Docker Desktop，然后重新运行此脚本
    echo.
    pause
    exit /b 1
)

REM 检查 Milvus 容器是否在运行
set MILVUS_RUNNING=0
docker ps --format "{{.Names}}" 2>nul | findstr "milvus-standalone" >nul 2>nul
if not errorlevel 1 set MILVUS_RUNNING=1

if "%MILVUS_RUNNING%"=="1" (
    echo [信息] Milvus 容器已在运行
) else (
    echo [信息] 正在启动 Milvus 容器（首次可能需要几分钟拉取镜像）...
    docker compose -f vector-database.yml up -d
    if errorlevel 1 (
        echo [错误] Docker 启动失败
        pause
        exit /b 1
    )
)

REM 等待 Milvus 健康检查通过（轮询，最多 3 分钟）
echo [信息] 等待 Milvus 就绪（最多 180 秒）...
set MILVUS_OK=0
for /L %%i in (1,1,36) do (
    curl -f -s http://localhost:9091/healthz >nul 2>nul
    if not errorlevel 1 (
        if "!MILVUS_OK!"=="0" (
            set MILVUS_OK=1
            echo [成功] Milvus 已就绪
        )
    )
    if "!MILVUS_OK!"=="1" goto :milvus_done
    timeout /t 5 /nobreak >nul
)
echo [警告] Milvus 未在 180 秒内就绪，将继续（RAG 功能可能不可用）

:milvus_done
echo.

REM ============================================
REM 步骤 5: 启动 CLS MCP 服务
REM ============================================
echo [5/%TOTAL_STEPS%] 启动 CLS MCP 服务...
start "CLS MCP Server" /min %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 /nobreak >nul
echo [成功] CLS MCP 服务已启动
echo.

REM ============================================
REM 步骤 6: 启动 Monitor MCP 服务
REM ============================================
echo [6/%TOTAL_STEPS%] 启动 Monitor MCP 服务...
start "Monitor MCP Server" /min %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 /nobreak >nul
echo [成功] Monitor MCP 服务已启动
echo.

REM ============================================
REM 步骤 7: 启动 FastAPI 服务
REM ============================================
echo [7/%TOTAL_STEPS%] 启动 FastAPI 服务...

REM 检查端口 9900 是否被占用
netstat -ano 2>nul | findstr ":9900" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 (
    echo [警告] 端口 9900 已被占用，正在释放...
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":9900" ^| findstr "LISTENING"') do (
        taskkill /PID %%a /F >nul 2>nul
    )
    timeout /t 2 /nobreak >nul
)

start "SuperBizAgent API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
echo [信息] 等待 FastAPI 就绪（最多 30 秒）...

set API_OK=0
for /L %%i in (1,1,15) do (
    curl -s -o nul http://localhost:9900/health 2>nul
    if not errorlevel 1 (
        if "!API_OK!"=="0" (
            set API_OK=1
            echo [成功] FastAPI 服务已就绪
        )
    )
    if "!API_OK!"=="1" goto :api_ready
    timeout /t 2 /nobreak >nul
)

echo [警告] FastAPI 未在 30 秒内就绪
echo [提示] 请手动运行查看错误：
echo        %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
goto :api_done

:api_ready
echo.

REM ============================================
REM 步骤 8: 上传文档并打开浏览器
REM ============================================
echo [8/%TOTAL_STEPS%] 上传文档到向量数据库...
for %%f in (aiops-docs\*.md) do (
    echo   上传: %%~nxf
    curl -s -X POST http://localhost:9900/api/upload -F "file=@%%f" >nul 2>nul
)
echo [成功] 文档上传完成
echo.
echo [信息] 正在打开浏览器...
start "" http://localhost:9900

:api_done
echo.
echo ====================================
echo 服务启动完成！
echo ====================================
echo Web 界面: http://localhost:9900
echo API 文档: http://localhost:9900/docs
echo.
echo 查看日志:
echo   - FastAPI: logs\app_*.log
echo   - CLS MCP: type mcp_cls.log
echo   - Monitor: type mcp_monitor.log
echo 停止服务: stop-windows.bat
echo ====================================
echo.
echo 按任意键关闭此窗口（不影响已启动的服务）
pause >nul
