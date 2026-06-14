# ================= 1. 缓存准备阶段 =================
FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04 AS builder

# 引入 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin

WORKDIR /app

# 仅仅把依赖锁定文件放进来，利用缓存层
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# ================= 2. 最终运行阶段 =================
FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04

ENV DOCKER_ENV=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # 把接下来当场生成的虚拟环境路径加入全局变量
    PATH="/app/.venv/bin:$PATH"

# 安装轻量级 Python 核心（提供最基础的 C 库和自解释器）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-minimal \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. 把 uv 工具直接拿过来用
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin

# 2. 把第一阶段下载好的项目描述和缓存拷过来
COPY --from=builder /app /app

# 3. 关键点：在当前系统环境下，当场同步虚拟环境（因为有缓存，这一步在 1 秒内完成，且绝对匹配当前系统底层）
RUN uv sync --frozen --no-install-project

# 4. 拷贝你的业务代码
COPY main.py train_mnist.py utils.py ./
COPY frontend/ ./frontend/

# 5. 编写启动脚本
RUN echo '#!/bin/bash\n\
if [ ! -f "data/models/mnist_cnn.pth" ]; then\n\
  echo "==> 未检测到训练模型，正在启动 MNIST 训练..."\n\
  uv run python train_mnist.py\n\
else\n\
  echo "==> 检测到已有模型，跳过训练步骤。"\n\
fi\n\
echo "==> 正在启动主应用服务..."\n\
exec uv run python main.py' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]