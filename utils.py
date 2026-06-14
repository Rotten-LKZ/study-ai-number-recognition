from PIL import Image, ImageOps, ImageStat, ImageFilter
import os
import io
import uuid
import base64
import random

SAVE_PATH = 'data/images'
os.makedirs(SAVE_PATH, exist_ok=True)
def load_and_save_image_from_base64(base64_str: str) -> tuple[Image.Image, str]:
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("L")
    # 自适应反转：MNIST 是黑底白字，如果上传的是白底黑字则反转极性
    if ImageStat.Stat(img).mean[0] > 127:
        img = ImageOps.invert(img)
    
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        img = img.filter(ImageFilter.MaxFilter(3))
        max_size = max(img.width, img.height)
        square_img = Image.new("L", (max_size, max_size), 0)
        square_img.paste(img, ((max_size - img.width) // 2, (max_size - img.height) // 2))
        img = square_img
        padding = int(max_size * 0.1)
        if padding > 0:
            padding_size = (max_size + padding * 2, max_size + padding * 2)
            padded_img = Image.new("L", padding_size, 0)
            padded_img.paste(img, (padding, padding))
            img = padded_img
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    filename = f"{uuid.uuid4().hex}.png"
    img.save(os.path.join(SAVE_PATH, filename))
    return img, filename

def get_image_path(filename: str) -> str:
    return os.path.join(SAVE_PATH, filename)


def generate_question() -> tuple[str, int]:
    """
    生成四则运算算数问题，并且保证答案为 0-9 整数
    返回一个字符串形式的算数问题和一个整数形式的答案
    """
    operator = random.choice(['+', '-', '*', '/'])
    final_answer = random.randint(0, 9)
    if operator == '+':
        a = random.randint(-100, final_answer)
        b = final_answer - a
        question = f"{a} + {b}"
    elif operator == '-':
        a = random.randint(final_answer, 100)
        b = a - final_answer
        question = f"{a} - {b}"
    elif operator == '*':
        if final_answer == 0:
            a = random.randint(-100, 100)
            b = 0
        else:
            factors = [i for i in range(1, final_answer + 1) if final_answer % i == 0]
            a = random.choice(factors)
            b = final_answer // a
        if random.choice([True, False]):
            a, b = b, a
        if random.choice([True, False]):
            a = -a
            b = -b
        question = f"{a} * {b}"
    else:  # operator == '/'
        b = random.randint(1, 9)
        a = final_answer * b
        if random.choice([True, False]):
            a = -a
            b = -b
        question = f"{a} ÷ {b}"
    return question, final_answer
