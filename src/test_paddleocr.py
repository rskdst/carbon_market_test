import os
import random
import string

import requests
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

from setup_logging import setup_logging

logging = setup_logging("image-fields")


def __process_image(local_image_path):
    paddleOCR = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_space_char=True,
        show_log=False,
        use_gpu=True,
        ir_optim=True,
    )
    # img = cv2.imread(local_image_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imwrite("thresh.png", thresh)
    result = paddleOCR.ocr(local_image_path, cls=True)
    return result[0]


def extract_fields_from_image(img_path, field_text_position_dict):
    def save_image_from_url(url, save_path):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)

    def generate_random_filename():
        return "".join(random.choices(string.ascii_lowercase, k=10))

    local_saved_path = img_path
    is_fetched_from_url = False
    if img_path.startswith(("http://", "https://")):
        local_saved_path = f"./images/tmp/{generate_random_filename()}.png"
        save_image_from_url(img_path, local_saved_path)
        is_fetched_from_url = True

    ocr_result = __process_image(local_saved_path)
    if is_fetched_from_url:
        logging.debug(f"Removing the fetched image: {local_saved_path}")
        os.remove(local_saved_path)
    if not ocr_result or not ocr_result[0]:
        raise ValueError("OCR failed to extract text from the image")

    # Extract texts and log the number of fields
    txts = [line[1][0] for line in ocr_result]
    logging.debug(f"Number of fields extracted: {len(txts)} for image: {img_path}")

    # Use a dictionary comprehension to create the result
    result = {
        field: txts[position - 1] if 0 < position <= len(txts) else None
        for field, position in field_text_position_dict.items()
    }
    return result


def draw_boxes(local_imgage_path):
    ocr_result = __process_image(local_imgage_path)
    image = Image.open(local_imgage_path).convert("RGB")

    # Extract boxes, texts, and scores
    boxes, txts, scores = zip(
        *[(line[0], line[1][0], line[1][1]) for line in ocr_result]
    )

    im_show = draw_ocr(image, boxes, txts, scores, font_path="./assets/simfang.ttf")
    im_show = Image.fromarray(im_show)
    result_name = (
        f"./images/{local_imgage_path.split('/')[-1].split('.')[0]}_result.jpg"
    )
    im_show.save(result_name)


def show_text_position(local_imgage_path):
    ocr_result = __process_image(local_imgage_path)
    txts = [line[1][0] for line in ocr_result]
    for i, txt in enumerate(txts):
        logging.info(f"Position {i+1}: {txt}")


if __name__ == "__main__":
    draw_boxes("./images/deal.png")
