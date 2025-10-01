from PIL import Image
import numpy as np

img = Image.open("image.png")

gray_img = img.convert("L")

gray_array = np.array(gray_img)

print(gray_array.shape)