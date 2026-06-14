FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04

ENV DOCKER_ENV=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-minimal \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin

COPY --from=builder /app /app

RUN uv sync --frozen --no-install-project

COPY main.py train_mnist.py utils.py ./
COPY frontend/ ./frontend/

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