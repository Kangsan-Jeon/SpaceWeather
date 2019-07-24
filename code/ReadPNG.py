import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# img = Image.open("C:/EIT_284/train/log_1.e4/EIT_284_20000101_130605.png")
# im = Image.open("C:/Users/tks02/OneDrive/문서/우주환경/Histogram.png")

img = mpimg.imread("C:/EIT_284/train/power_1.5e3/EIT_284_20000101_130605.png")
# im = mpimg.imread("C:/Users/tks02/OneDrive/문서/우주환경/Histogram.png")

print(img, '\n')

print(np.max(img.flatten()), np.min(img.flatten()))
# print(np.max(im.flatten()), np.min(im.flatten()))

plt.figure("img")
plt.imshow(img)
plt.show()
