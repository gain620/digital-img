import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def histogram_equalize(input_img):
    # result img
    output_img = np.zeros(img.shape, dtype=np.int16)

    # for normalized sums of a result img
    norm_value = np.zeros(256, dtype=np.float16)

    hist, bins = np.histogram(input_img.flatten(), 256, [0, 256])

    # cumulative sums per pixel
    cdf = hist.cumsum()

    # len(hist) = 256 , cdf[-1] = 800x800 (num of pixels)
    for i in range(0, len(hist)):
        norm_value[i] = hist[i] / float(cdf[-1])

    # final normalized sums
    new_cdf = norm_value.cumsum()

    for i in range(0, input_img.shape[0]):
        for j in range(0, input_img.shape[1]):
            output_img[i][j] = int(round(new_cdf[img[i][j]] * 255))

    # limit pixels to greyscale from 0 ~ 255
    for i in range(0, input_img.shape[0]):
        for j in range(0, input_img.shape[1]):
            if output_img[i][j] < 0:
                output_img[i][j] = 0
            elif output_img[i][j] > 256:
                output_img[i][j] = 256

    return output_img


'''
print(img.shape)
print(cdf[-1])
print(hist)
print(res_value)
print(res_img)
print(img[30, 30])
'''

# start equalizing input image
img = misc.imread('./sample.jpg')
new_img = histogram_equalize(img)

plt.imshow(new_img, cmap = 'gray')
plt.show()
