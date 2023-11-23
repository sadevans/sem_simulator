import numpy as np

import cv2
import matplotlib.pyplot as plt

import scipy
from scipy.interpolate import CubicSpline, lagrange
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
import random
import os
from skimage.draw import line
from scipy.ndimage import gaussian_filter1d

import torch
from torch.distributions.gamma import Gamma

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

from all import *
from transform import *
from draw import *



if __name__ == '__main__':
    img_circles = simulate_circles()
    cv2.imwrite('difr_circles.png', np.clip(img_circles, 0, 255))

    ext, int, img_circles = detect_contours(img_circles)
    width_img= transform(img_circles, ext, int)

    plt.imshow(width_img)