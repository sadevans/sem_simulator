import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline, lagrange
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
import random




if __name__ == '__main__':
    
    bin_mask = cv2.imread('')