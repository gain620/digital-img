import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def limit(img_arr):
    for i in range(0, 256):
        for j in range(0, 256):
            if img_arr[i,j] < 0 :
                img_arr[i, j] = 0
            elif img_arr[i,j] > 256:
                img_arr[i, j] = 256


img = misc.imread('./sample.jpg')

# result img
res_img = np.zeros(img.shape, dtype=np.int16)

# normalized sum of result img
res_value = np.zeros(256, dtype=np.float16)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Cumulative sum per pixel
cdf = hist.cumsum()


for i in range(0, 256):
    res_value[i] = hist[i] / float(cdf[-1])

new_cdf = res_value.cumsum()


for i in range(0, 800):
    for j in range(0, 800):
        res_img[i,j] = int(round(new_cdf[img[i,j]] * 255))

limit(res_img)

print(img.shape)
print(cdf[-1])
print(hist)
print(res_value)
print(res_img)
print(img[30,30])

plt.imshow(res_img, cmap = 'gray')
plt.show()