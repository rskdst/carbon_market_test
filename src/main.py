import argparse
import csv
import datetime
import functools
import os
import random
import re
import string
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from urllib.parse import urljoin

import oss2
import requests
from bs4 import BeautifulSoup
from paddleocr import PaddleOCR
from retrying import retry
from img2table.document import Image
from img2table.ocr import PaddleOCR as Img2TablePaddleOCR


from setup_logging import setup_logging

logging = setup_logging("main")

URL = "https://www.cneeex.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


PRICE_IMAGE_POSITION = {"CEA_listed_Price_increases": 20}


DEAL_IMAGE_POSITION = {
    "CEA22_Trading_volume": (9, 6),
    "CEA22_listed_Trading_volume": (7, 6),
    "CEA22_block_Trading_volume": (8, 6),
    "CEA22_listed_Closing_price": (8, 4),
    "CEA22_listed_Price_increases": (8, 5),
    "CEA22_listed_Trading_amount": (7, 7),
    "CEA22_block_Trading_amount": (8, 7),
    "CEA21_Trading_volume": (6, 6),
    "CEA21_listed_Trading_volume": (4, 6),
    "CEA21_block_Trading_volume": (5, 6),
    "CEA21_listed_Closing_price": (5, 4),
    "CEA21_listed_Price_increases": (5, 5),
    "CEA21_listed_Trading_amount": (4, 7),
    "CEA21_block_Trading_amount": (5, 7),
    "CEA1920_Trading_volume": (3, 6),
    "CEA1920_listed_Trading_volume": (1, 6),
    "CEA1920_block_Trading_volume": (2, 6),
    "CEA1920_listed_Closing_price": (2, 4),
    "CEA1920_listed_Price_increases": (2, 5),
    "CEA1920_listed_Trading_amount": (1, 7),
    "CEA1920_block_Trading_amount": (2, 7),
}

IMAGE_FIELD_CALCULATE = {
    "CEA1920_block_price":"CEA1920_block_Trading_amount/CEA1920_block_Trading_volume",
    "CEA21_block_price":"CEA21_block_Trading_amount/CEA21_block_Trading_volume",
    "CEA22_block_price":"CEA22_block_Trading_amount/CEA22_block_Trading_volume",
}

COMMON_REGEX_PATTERN = "\d{1,3}(?:,\d{3})*(?:\.\d+)?"

TEXT_FIELD_PATTERN = [
    r"全国碳市场每日综合价格行情及成交信息(?P<date>\d{8})",
    rf"今日全国碳排放配额总成交量(?P<CEA_Trading_volume>{COMMON_REGEX_PATTERN})吨",
    rf"今日挂牌协议交易成交量(?P<CEA_listed_Trading_volume>{COMMON_REGEX_PATTERN})吨",
    rf"大宗协议交易成交量(?P<CEA_block_Trading_volume>{COMMON_REGEX_PATTERN})吨",
    rf"收盘价(?P<CEA_Closing_price>{COMMON_REGEX_PATTERN})元/吨",
    rf"今日挂牌协议交易成交量{COMMON_REGEX_PATTERN}吨，成交额(?P<CEA_listed_Trading_amount>{COMMON_REGEX_PATTERN})元",
    rf"大宗协议交易成交量{COMMON_REGEX_PATTERN}吨，成交额(?P<CEA_block_Trading_amount>{COMMON_REGEX_PATTERN})元。",
    rf"全国碳市场碳排放配额累计成交量(?P<Total_trading_volume_of_carbon_market>{COMMON_REGEX_PATTERN})吨，累计成交额(?P<CNY_amount_of_carbon_market>{COMMON_REGEX_PATTERN})元",
]

TEXT_FIELD_CALCULATE = {
    "CEA_block_price":"CEA_block_Trading_amount/CEA_block_Trading_volume",
    "CEA_listed_price":"CEA_listed_Trading_amount/CEA_listed_Trading_volume",
    "CEA_total_price":"(CEA_listed_Trading_amount+CEA_block_Trading_amount)/CEA_Trading_volume"
}

def timing_decorator(func, is_param_logging=False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if is_param_logging:
            logging.info(
                f"{func.__name__} took {elapsed_time:.4f} seconds to run with parameters: {args}, {kwargs}"
            )
        else:
            logging.info(f"{func.__name__} took {elapsed_time:.4f} seconds to run.")
        return result

    return wrapper


def fetch_html(url):
    """Fetch HTML content from the given URL."""
    logging.debug(f"Fetching HTML content from URL: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching HTML from {url}: {e}")
        return None


def parse_html(html_content):
    """Parse HTML content using BeautifulSoup."""
    return BeautifulSoup(html_content, "html.parser")


def extract_fields_from_image(img_path, field_text_position_dict, img2tableFlag=False):
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

    def __img2table(local_image_path):
        ocr = Img2TablePaddleOCR(lang="ch")
        img = Image(local_image_path, detect_rotation=False)
        tables = img.extract_tables(ocr=ocr, borderless_tables=True)
        df = tables[0].df
        return df.values.tolist()

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

    ocr_result = (
        __img2table(local_saved_path)
        if img2tableFlag
        else __process_image(local_saved_path)
    )
    if is_fetched_from_url:
        logging.debug(f"Removing the fetched image: {local_saved_path}")
        os.remove(local_saved_path)
    if not ocr_result or not ocr_result[0]:
        raise ValueError("OCR failed to extract text from the image")

    # Extract texts and log the number of fields
    txts = ocr_result if img2tableFlag else [line[1][0] for line in ocr_result]
    logging.debug(f"Number of fields extracted: {len(txts)} for image: {img_path}")

    # Use a dictionary comprehension to create the result
    if img2tableFlag:
        result = {
            field: (
                txts[position[0]][position[1]] if 0 < position[0] <= len(txts) else None
            )
            for field, position in field_text_position_dict.items()
        }
    else:
        result = {
            field: (txts[position - 1] if 0 < position <= len(txts) else None)
            for field, position in field_text_position_dict.items()
        }
    return result


def extract_image_fields(soup, link):
    """Extract and log image sources with titles."""
    img_elements = soup.find_all("img", title=True)
    valid_sources = [img["src"] for img in img_elements if "upload" in img["src"]]
    result = {}
    if len(valid_sources) != 2:
        logging.warning(
            f"Expected 2 valid image sources, but found {len(valid_sources)}. Skipping image extraction."
        )
        return result
    decorated_extract_fields_from_image = timing_decorator(
        extract_fields_from_image, is_param_logging=True
    )
    price_image_result = decorated_extract_fields_from_image(
        construct_full_url(URL, valid_sources[0]), PRICE_IMAGE_POSITION
    )
    result.update(price_image_result)

    deal_image_result = decorated_extract_fields_from_image(
        construct_full_url(URL, valid_sources[1]), DEAL_IMAGE_POSITION, True
    )
    data_float = {}
    for key,value in deal_image_result.items():
        try:
            data_float[key] = eval(value.replace(',', ''))
        except:
            continue
    for field_name, expression in IMAGE_FIELD_CALCULATE.items():
        try:
            calculated_value = eval(expression, {}, data_float)
            deal_image_result[field_name] = calculated_value
        except Exception:
            deal_image_result[field_name] = None
    result.update(deal_image_result)
    return result


def extract_text_fields(soup, link):
    extracted_data = {}
    text_content = soup.get_text()

    for pattern in TEXT_FIELD_PATTERN:
        compiled_pattern = re.compile(pattern)
        group_names = compiled_pattern.groupindex.keys()
        match = compiled_pattern.search(text_content)
        if match:
            extracted_data.update(match.groupdict())
        else:
            logging.warning(
                f"Parameters: {group_names} not found in text content from link: {link}."
            )
            extracted_data.update({key: None for key in group_names})
    data_float = {key: eval(value.replace(',','')) for key, value in extracted_data.items()}
    for field_name,expression in TEXT_FIELD_CALCULATE.items():
        try:
            calculated_value = eval(expression, {}, data_float)
            extracted_data[field_name] = calculated_value
        except Exception:
            extracted_data[field_name] = None
    return extracted_data


def process_daily_data(link, text):
    logging.info(f"Processing daily data for: {text} from {link}.")
    html_content = fetch_html(link)
    result = {}
    if html_content:
        result["created_on"] = datetime.datetime.now().isoformat()
        result["data_source_link"] = link
        soup = parse_html(html_content)
        result.update(
            {k.lower(): v for k, v in extract_text_fields(soup, link).items()}
        )
        result.update(
            {k.lower(): v for k, v in extract_image_fields(soup, link).items()}
        )
    return result


def get_all_page_links(year):
    year_link = f"https://www.cneeex.com/qgtpfqjy/mrgk/{year}n/"
    result = {1: year_link}
    requested_urls = set([year_link])
    max_page_num = 1

    def process_links(current_url):
        nonlocal max_page_num
        link_text_tuples = extract_page_links(current_url)
        next_request_url = None

        for page_num_str, page_link in link_text_tuples:
            page_num = int(page_num_str)
            result[page_num] = page_link
            if page_num > max_page_num:
                max_page_num = page_num
                next_request_url = page_link

        return next_request_url

    next_url = process_links(year_link)
    while next_url and next_url not in requested_urls:
        requested_urls.add(next_url)
        next_url = process_links(next_url)

    return result


def extract_page_links(year_link):
    html_content = fetch_html(year_link)
    if html_content:
        soup = parse_html(html_content)
        z_num_elements = soup.find_all("a", class_="z_num")
        return [
            (element.text, construct_full_url(year_link, element["href"]))
            for element in z_num_elements
        ]
    return []


def construct_full_url(base_url, relative_url):
    return (
        relative_url
        if relative_url.startswith(("http://", "https://"))
        else urljoin(base_url, relative_url)
    )


def process_page_data(page_link, executor: ThreadPoolExecutor):
    html_content = fetch_html(page_link)
    result = []
    if html_content:
        soup = parse_html(html_content)
        li_elements = soup.find_all("li", class_="text-ellipsis hidden-xs")
        daily_data = [
            (construct_full_url(page_link, li.find("a")["href"]), li.find("a").text)
            for li in li_elements
        ]

        futures = []
        for full_url, text in daily_data:
            future = executor.submit(process_daily_data, full_url, text)
            futures.append(future)

        for future in futures:
            try:
                data = future.result()
                if data:
                    result.append(data)
                    logging.debug(f"Processed data: {data}")
                else:
                    logging.warning("No data processed for this entry")
            except Exception as e:
                logging.error(f"Error processing data: {str(e)}")

    return result


def process_year_data(year, func):
    all_page_links = get_all_page_links(year)
    result = []
    logging.info(f"Found {len(all_page_links)} pages for year {year}.")
    core_num = os.cpu_count()
    logging.info(f"Using {core_num} cores for data processing.")
    executor = ThreadPoolExecutor(
        max_workers=core_num, thread_name_prefix="DataProcessThreadPool"
    )
    try:
        for page_num, page_link in all_page_links.items():
            logging.info(f"Processing page {page_num}{'-' * 50}")
            start_time = time.time()
            page_result = process_page_data(page_link, executor)
            func(page_result)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(
                f"Found {len(page_result)} entries in page {page_num}. Time elapsed: {elapsed_time} seconds."
            )
            result.extend(page_result)
            logging.info(f"Finished processing page {page_num}{'-' * 50}")
        return result
    finally:
        executor.shutdown()


@retry(stop_max_attempt_number=3, wait_fixed=300000)  # 5 分钟
def process_current_day_data():
    now = datetime.datetime.now() - timedelta(days=1)
    year = now.year
    current_day = now.strftime("%Y%m%d")
    year_link = f"https://www.cneeex.com/qgtpfqjy/mrgk/{year}n/"
    html_content = fetch_html(year_link)
    if html_content:
        soup = parse_html(html_content)
        li_elements = soup.find_all("li", class_="text-ellipsis hidden-xs")
        daily_data = [
            (construct_full_url(year_link, li.find("a")["href"]), li.find("a").text)
            for li in li_elements
        ]
        first_link, first_text = daily_data[0]
        if current_day in first_text:
            return process_daily_data(first_link, first_text)
        else:
            logging.warning(f"Expected data for {current_day}, but found {first_text}.")
            raise ValueError("Data not found for current day.")


def write_result_to_csv(result, filename):
    fieldnames = list(result[0].keys()) if result else []
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            logging.info(f"Writing header: {fieldnames} to file: {filename}")
            writer.writeheader()
        writer.writerows(result)


@retry(
    stop_max_attempt_number=3,
    wait_random_min=500,
    wait_random_max=1500,
)
def upload_to_oss(
    local_file_path,
    oss_access_key_id,
    oss_access_key_secret,
    bucket_name,
    endpoint,
    base_path,
):
    auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d")
    filename = os.path.basename(local_file_path)
    if base_path:
        object_key = os.path.join(base_path, date, filename)
    else:
        object_key = os.path.join(date, filename)
    bucket.put_object_from_file(object_key, local_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("CARBON_RUN_MODE", "day"),
        choices=["year", "day"],
        help="Specify the mode of data processing.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=int(os.environ.get("CARBON_DATA_YEAR", datetime.datetime.now().year)),
        help="Process data for a specific year (default: current year)",
    )
    parser.add_argument(
        "--oss-access-key-id",
        type=str,
        default=os.environ.get("CARBON_OSS_ACCESS_KEY_ID"),
        help="The access key ID for the OSS service.",
    )

    parser.add_argument(
        "--oss-access-key-secret",
        type=str,
        default=os.environ.get("CARBON_OSS_ACCESS_KEY_SECRET"),
        help="The access key secret for the OSS service.",
    )

    parser.add_argument(
        "--oss-bucket_name",
        type=str,
        default=os.environ.get("CARBON_OSS_BUCKET_NAME"),
        help="The name of the bucket to upload the file to.",
    )
    parser.add_argument(
        "--oss-endpoint",
        type=str,
        default=os.environ.get("CARBON_OSS_ENDPOINT"),
        help="The endpoint of the OSS service.",
    )

    parser.add_argument(
        "--oss-base-path",
        type=str,
        default=os.environ.get("CARBON_OSS_BASE_PATH"),
        help="The base path to store the file in the OSS bucket.",
    )
    args = parser.parse_args()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    filename = f"./results/data_{timestamp}.csv"
    if args.mode == "year":
        write_result_func = functools.partial(write_result_to_csv, filename=filename)
        result = process_year_data(args.year, write_result_func)
    else:
        result = [process_current_day_data()]
        write_result_to_csv(result, filename)

    # upload_to_oss(
    #     filename,
    #     args.oss_access_key_id,
    #     args.oss_access_key_secret,
    #     args.oss_bucket_name,
    #     args.oss_endpoint,
    #     args.oss_base_path,
    # )
