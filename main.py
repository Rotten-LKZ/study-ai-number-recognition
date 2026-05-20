from flask import Flask, request, send_file, make_response, Blueprint
from flask_sqlalchemy import SQLAlchemy
from passlib.context import CryptContext
from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from functools import wraps
from train_mnist import CNN, DEVICE
from torchvision import transforms
from utils import load_and_save_image_from_base64, get_image_path
import torch
import jwt
import datetime
import os

class Base(DeclarativeBase):
    pass

# 用户表，存储注册用户的用户名和密码（Argon2）
class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(200), nullable=False)

# 识别历史表，记录用户每次上传的图片和识别结果
class RecognitionHistory(Base):
    __tablename__ = 'recognition_history'

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column()
    image_filename: Mapped[str] = mapped_column(nullable=False)
    results: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column()

db = SQLAlchemy(model_class=Base)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
app = Flask(__name__)
# 模型初始化
model_path = "data/models/mnist_cnn.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行 uv run python train_mnist.py 训练")
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
# 数据库初始化
os.makedirs('data/db', exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/db/app.db')}'
app.config['SECRET_KEY'] = 'MNIST_Rec0gniti0n_Secret_Key_F0r_Dev_Only_Very_Long'
# 创建数据库表
db.init_app(app)
with app.app_context():
    db.create_all()

# 检验请求 JSON 是否包含必要的字段的装饰器
def validate_json(keys):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            if not data:
                return {"message": "请求必须包含JSON数据"}, 400
            for key in keys:
                if key not in data:
                    return {"message": f"缺少必要的参数: {key}"}, 400
            return f(*args, **kwargs)
        return wrapper
    return decorator

# 验证用户登录状态的装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return {"message": "Unauthorized"}, 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return {"message": "Token expired"}, 401
        except jwt.InvalidTokenError:
            return {"message": "Invalid token"}, 401
        return f(*args, **kwargs, user_id=current_user_id)
    return decorated

@app.route('/', methods=['GET'])
def index():
    return send_file('frontend/index.html')

# 服务静态文件 服务frontend下 默认尝试 html 后缀
@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    file_path = os.path.join('frontend', path)
    if os.path.exists(file_path):
        return send_file(file_path)
    html_path = os.path.join('frontend', f'{path}.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    return {"message": "Not found"}, 404

# API 蓝图
api_bp = Blueprint('api', __name__)

@api_bp.route('/register', methods=['POST'])
@validate_json(['username', 'password'])
def register():
    data = request.get_json()

    hashed_password = pwd_context.hash(data['password'])
    new_user = User(username=data['username'], password=hashed_password)
    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception:
        db.session.rollback()
        return {"message": "Registration failed"}, 400
    return {"message": "Register successful"}, 201

@api_bp.route('/login', methods=['POST'])
@validate_json(['username', 'password'])
def login():
    data = request.get_json()
    user = db.session.execute(select(User).where(User.username == data.get('username')).limit(1)).scalar_one_or_none()
    if user and pwd_context.verify(data.get('password'), user.password):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=10)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        resp = make_response({"message": "Login successful"})
        resp.set_cookie('token', token, httponly=True, samesite='Lax')
        return resp
        
    return {"message": "Invalid credentials"}, 401

@api_bp.route('/me', methods=['GET'])
@token_required
def me(user_id):
    user = db.session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not user:
        return {"message": "Unauthorized"}, 401
    return {"username": user.username}

@api_bp.route('/logout', methods=['POST'])
def logout():
    resp = make_response({"message": "Logout successful"})
    resp.delete_cookie('token')
    return resp

@api_bp.route('/recognize', methods=['POST'])
@validate_json(['image']) # in BASE64
@token_required
def recognize(user_id):
    img, image_path = load_and_save_image_from_base64(request.get_json()['image'])
    tensor = transforms.ToTensor()(img)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        topk = torch.topk(probs, 3, dim=1)

    results = [(int(idx.item()), float(val.item())) for val, idx in zip(topk.values[0], topk.indices[0])]
    history_item = RecognitionHistory(user_id=user_id, image_filename=os.path.basename(image_path), results=str(results), created_at=datetime.datetime.now())
    db.session.add(history_item)
    db.session.commit()
    return {"message": "Recognition successful", "result": results}


@api_bp.route('/recognition_history', methods=['GET'])
@token_required
def recognition_history(user_id):
    history = db.session.execute(select(RecognitionHistory).where(RecognitionHistory.user_id == user_id)).scalars()
    return {"history": [{"image_filename": item.image_filename, "results": list(item.results), "created_at": item.created_at} for item in history]}

@api_bp.route('/images/<filename>', methods=['GET'])
@token_required
def get_image(filename, user_id):
    image_path = get_image_path(filename)
    if not os.path.exists(image_path):
        return {"message": "Image not found"}, 404
    return send_file(image_path, mimetype='image/png')

app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)