import numpy as np
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

print(os.getcwd())
print("Hello")

image = Image.open(r'C:\Users\paulcand94\Downloads\STL10\stl10_raw\test\airplane\airplane_test_001.png')

image.show()

gray_image = image.convert("L")

gray_image.show()

im = np.asarray(gray_image)
print(im.shape[0], im.shape[1])
