import numpy as np

from ..implementations.convolution import convolve2d

zeros = np.zeros((10,5))
ones = np.ones((10,5))

img = np.hstack((zeros,ones))

print(img)
print(img.shape)

kernel = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

feature_map = convolve2d(img, kernel, stride=2,padding=1)
print(feature_map)
print(feature_map.shape)