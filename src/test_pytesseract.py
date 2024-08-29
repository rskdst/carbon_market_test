# 安装所需库
import cv2
import pandas as pd
import pytesseract

# 读取包含表格的图片
img = cv2.imread("./images/deal.png")

# 对图片进行预处理,如灰度化、二值化等,提高识别准确率
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# 使用pytesseract识别图片中的文字
text = pytesseract.image_to_string(thresh)

print(text)
# 将识别结果按行分割成列表
lines = text.split("\n")

# 创建一个空列表用于存储表格数据
data = []

# 遍历每一行
for line in lines:
    # 按空格或制表符分割每一行,提取单元格数据
    cells = line.split()
    data.append(cells)

# 将提取的数据转换为pandas的DataFrame对象
df = pd.DataFrame(data)

# 清理数据,删除空行等
df.dropna(how="all", inplace=True)

# 保存为Excel文件
df.to_excel("results/output.xlsx", index=False)
