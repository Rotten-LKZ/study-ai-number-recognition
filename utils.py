from PIL import Image, ImageOps, ImageStat
import os
import io
import uuid
import base64

SAVE_PATH = 'data/images'
os.makedirs(SAVE_PATH, exist_ok=True)
def load_and_save_image_from_base64(base64_str: str) -> tuple[Image.Image, str]:
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("L").resize((28, 28))
    # 自适应反转：MNIST 是黑底白字，如果上传的是白底黑字则反转极性
    if ImageStat.Stat(img).mean[0] > 127:
        img = ImageOps.invert(img)
    filename = f"{uuid.uuid4().hex}.png"
    img.save(os.path.join(SAVE_PATH, filename))
    return img, filename

def get_image_path(filename: str) -> str:
    return os.path.join(SAVE_PATH, filename)