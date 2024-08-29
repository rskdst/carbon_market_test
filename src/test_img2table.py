from img2table.document import Image
from img2table.ocr import PaddleOCR

ocr = PaddleOCR(lang="ch")

src = "./images/deal.png"
img = Image(src, detect_rotation=False)
tables = img.extract_tables(ocr=ocr, borderless_tables=True)
df = tables[0].df
df.to_excel("./results/deal.xlsx", index=False)

src = "./images/deal2.png"
img = Image(src, detect_rotation=False)
tables = img.extract_tables(ocr=ocr, borderless_tables=True)
df = tables[0].df
df.to_excel("./results/deal2.xlsx", index=False)
