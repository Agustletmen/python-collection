import tesserocr
from PIL import Image

image = Image.open(f"C:\\Users\\15272\\Desktop\\v6.png")
print(image)  # 打印图片信息
res = tesserocr.image_to_text(image, lang='chi_sim', psm=8)
# res = tesserocr.image_to_text(image, lang='eng', psm=8)
print(res)
