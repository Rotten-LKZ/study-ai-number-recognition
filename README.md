# 手写数字识别

基于 PyTorch CNN + Flask 的手写数字识别 Web 应用。用户可在网页画布上手写数字，后端通过深度学习模型实时识别，并支持用户注册登录与历史记录查询。

## 快速开始

### 1. 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器

### 2. 安装依赖

```bash
uv sync
```

### 3. 训练模型（首次运行）

```bash
uv run python train_mnist.py
```

> 脚本会自动下载 MNIST 数据集，训练完成后模型保存至 `data/models/mnist_cnn.pth`。

### 4. 启动服务

```bash
uv run python main.py
```

### 5. 访问应用

浏览器打开 http://localhost:5000

## 使用 Docker 直接部署
### 1. 创建 `docker-compose.yml` 文件并且填写类似内容
```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: ghcr.io/rotten-lkz/study-ai-number-recognition:latest
    container_name: python-mnist-app
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```
### 2. 运行 `docker compose up -d`，如果没有已有的模型文件会自动训练，之后便会自动启动应用

### 说明

该 Docker 镜像由 GitHub Actions 自动构建，触发条件是带 v* tag 的 commit。

---

## API 概览

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | /api/register | 用户注册 | 否 |
| POST | /api/login | 用户登录（返回 JWT Cookie） | 否 |
| POST | /api/logout | 用户登出 | 否 |
| POST | /api/login_on_phone | 生成临时 JWT | 是 |
| GET | /api/login_by_temporary_token | 用临时 JWT 登录 | 否 |
| GET | /api/me | 获取当前用户信息 | 是 |
| POST | /api/recognize | 上传 Base64 图片进行识别 | 是 |
| GET | /api/recognition_history | 获取识别历史记录 | 是 |
| GET | /api/images/<filename> | 获取历史图片 | 是 |

---

## 模型架构

```
CNN(
  Conv2d(1, 32, 3) → ReLU → MaxPool2d(2)
  Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
  Flatten
  Linear(64*7*7, 128) → ReLU → Dropout(0.5)
  Linear(128, 10)
)
```

- 输入：28×28 灰度图像
- 输出：10 个数字类别的概率分布
- 优化器：Adam，学习率 0.001

---

## 注意事项

- `SECRET_KEY` 当前为开发环境硬编码，生产环境请替换为安全随机密钥
- 模型文件 `data/models/mnist_cnn.pth` 必须预先训练生成，否则服务启动会报错

## 版权声明

/frontend/resource 下使用了几个外部项目：
- [Bootstrap](https://github.com/twbs/bootstrap) under MIT
- [qrcode.js](https://github.com/davidshimjs/qrcodejs) under MIT
- [sweetalert2](https://github.com/sweetalert2/sweetalert2) under MIT
