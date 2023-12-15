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

from all import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transform(img, ext, int):
    color_back = 110
    color_hole = 85

    width_img = np.zeros_like(img, dtype=np.float32)
    angles_img = np.zeros_like(img, dtype=np.float32)
    new_angles = np.zeros_like(img, dtype=np.float32)
    color_map = np.zeros_like(img, dtype=np.float32)
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole


    for cont_ext, cont_int in zip(ext, int):
        for point in cont_ext:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_int)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_int[index] 
                    prev = [compute_previous_pixel(point, nearest_point)]
                    next = [compute_next_pixel(point, nearest_point)]

                    
                    # if nearest_point[0] == point[0] and nearest_point[1] == point[1]:
                    #     dist_ = 1
                    # else:
                        # dist_ = np.sqrt((point[0] - nearest_point[0])**2 + (point[1] - nearest_point[1])**2).astype(np.int32)
                    # dist_ = np.sqrt((prev[0][0] - nearest_point[0])**2 + (prev[0][1] - nearest_point[1])**2).astype(np.int32)
                    
                    # print(min_dist, dist_)
                    # dist_ = dist_.item()
                    discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
                    # discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
                    
                    dist_ = len(discrete_line)
                # print(next)

                angle = np.arctan(700 / (dist_ * 12))
                if dist_ <= 10: cv2.line(width_img, point, nearest_point, dist_, 3)
                if dist_ <= 10:cv2.line(angles_img, point, nearest_point, angle, 2)

                if len(discrete_line) == 1:
                    height_vals = [color_back - 2]
                    discrete_line_ = discrete_line

                else:
                    height_vals = [((k_ * ((color_back - color_hole) / dist_)) + color_hole) for k_ in range(0, dist_+2)]
                    discrete_line_ = discrete_line + next

                # if dist_ <= 10: draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
                # else: draw_gradient_line(color_map, point, discrete_line_, height_vals[-1::-1], thickness=1)
                draw_gradient_line(color_map, point, discrete_line_, height_vals[-1::-1], thickness=3)

                # calculate new angles
                if dist_ >= 3:

                    # firct variant

                    # if dist_ > 5:
                        k = dist_/10
                        # if k > 1:
                        # print(dist, k, angle)
                        # if k < 1:
                        #     val = angle*k
                        # else:
                        #     val = angle/k
                        val = angle * 0.9

                        y_mean = val
                        y_0 = val/2
                        y_n = val/2
                        x_plot = np.arange(0, len(discrete_line))
                        coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                        a,b,c = coefs[0]
                        y_parabola = parabola(x_plot, a, b, c)
                        heights = y_parabola
                        draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=2)
                    # else:
                    #     val = angle
                    #     cv2.line(new_angles, point, nearest_point, val, 4)
                    # val = np.arctan(700/(dist_+2))*0.9

                    # if dist_ <= 5:
                    #     k = dist_/10
                    #     val = np.abs(angle*(1-k) * 0.95)

                    #     # val = angle*(1-k) * 0.95
                    # else:
                    #     k = dist_/5
                    # val = np.abs(angle*(1-k) * 0.95)
                    # if k < 1:
                    #     val = angle*(1-k) * 0.95
                        # val = angle/k * 0.95

                    # else:
                        # val = angle/k * 0.95


                    # ___ added - плохо видно широкую границу 9-10 пикселей

                    # if dist_ > 6:
                        # dist_ = 6
                    # k = (dist_-2)/10
                    # val = angle * (1 - k)
                    # val = angle * (1 - k)

                    # another add
                    # val = 0.9 * angle

                    # added second - пытаюсь улучшить видимость широкой границы
                    # if dist_>=8:
                    #     val = val * 1.1
                    # print(dist_, angle, k, val)

                    # y_mean = val
                    # y_0 = val/2
                    # y_n = val/2
                    # y_0 = val/6
                    # y_n = val/6
                    # x_plot = np.arange(0, len(discrete_line)+2)
                    # x_plot = np.arange(0, len(discrete_line))

                    # x_plot = np.arange(0, len(discrete_line)+1)

                    # coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                    # coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2 - 1], x_plot[-1]], [y_0, y_mean, y_n])
                    # coefs = curve_fit(parabola, [x_plot[0], x_plot[1], x_plot[len(x_plot)//2 - 1], x_plot[-2], x_plot[-1]], [0, y_0, y_mean, y_n, 0])


                    # a,b,c = coefs[0]
                    # y_parabola = parabola(x_plot, a, b, c)
                    # y_parabola = blur(y_parabola)
                    # y_parabola =  gaussian_filter1d(y_parabola, 3)
                    # heights = y_parabola
                    # if dist_ > 10:
                    #     thickness = 5
                    # else:
                    #     thickness = 5
                    # discrete_line_ = prev + discrete_line + next
                    # discrete_line_ = discrete_line

                    # discrete_line_ = prev + discrete_line

                    # draw_gradient_line(new_angles, point, discrete_line_, y_parabola, thickness=thickness)


    for cont_ext, cont_int in zip(ext, int):
        for point in cont_int:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_ext)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_ext[index]
                    # prev = [compute_previous_pixel(point, nearest_point)]
                    next = [compute_next_pixel(point, nearest_point)]
                    prev = [compute_previous_pixel(point, nearest_point)]

                    discrete_line = list(zip(*line(*point, *next[0]))) # find all pixels from the line
                    dist_ = len(discrete_line)
                    # if nearest_point[0] == point[0] and nearest_point[1] == point[1]:
                    #     dist_ = 1
                    # else:
                        # dist_ = np.sqrt((point[0] - nearest_point[0])**2 + (point[1] - nearest_point[1])**2).astype(np.int32)
                    # dist_ = np.sqrt((prev[0][0] - nearest_point[0])**2 + (prev[0][1] - nearest_point[1])**2).astype(np.int32)
                    # dist_ = np.sqrt((next[0][0] - nearest_point[0])**2 + (next[0][1] - nearest_point[1])**2).astype(np.int32)
                    
                    # dist_ = dist_.item()

                # dist_ += 1
                if dist_ == 0:
                    dist_ = 1

                angle = np.arctan(700 / (dist_ * 12))
                cv2.line(width_img, point, nearest_point, dist_, 3)
                cv2.line(angles_img, point, nearest_point, angle, 2)

                if len(discrete_line) == 1:
                    height_vals = [color_back - 2]
                    discrete_line_ =discrete_line

                else:
                    height_vals = [((k_ * ((color_back - color_hole) / dist_)) + color_hole) for k_ in range(1, dist_+2)]
                    discrete_line_ = discrete_line + next

                # discrete_line_ =discrete_line + next
                # draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
                # if dist_ <= 10:draw_gradient_line(color_map, point, discrete_line_, height_vals[-1::-1], thickness=3)
                if dist_ > 10:
                    if point[0] != nearest_point[0] and point[1] != nearest_point[1]:
                        draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
                        
                # if dist_ >= 3:


                    # firct variant

                    # if dist_ > 5:
                    #     k = dist_/10
                    #     # if k > 1:
                    #     # print(dist, k, angle)
                    #     if k < 1:
                    #         val = angle*k
                    #     else:
                    #         val = angle/k

                        val = angle * 0.9
                        y_mean = val
                        y_0 = val/2
                        y_n = val/2
                        x_plot = np.arange(0, len(discrete_line))
                        coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                        a,b,c = coefs[0]
                        y_parabola = parabola(x_plot, a, b, c)
                        heights = y_parabola
                        # print(y_parabola)
                        draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=2)
                if dist_ <= 10:
                    draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
                     
                # calculate new angles
                if dist_ >= 3 and dist_ <= 10:
                        
                # if dist_ >= 3:


                    # firct variant

                    # if dist_ > 5:
                    #     k = dist_/10
                    #     # if k > 1:
                    #     # print(dist, k, angle)
                    #     if k < 1:
                    #         val = angle*k
                    #     else:
                    #         val = angle/k

                        val = angle * 0.9
                        y_mean = val
                        y_0 = val/2
                        y_n = val/2
                        x_plot = np.arange(0, len(discrete_line))
                        coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                        a,b,c = coefs[0]
                        y_parabola = parabola(x_plot, a, b, c)
                        heights = y_parabola
                        # print(y_parabola)
                        draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=2)
                    # else:
                    #     val = angle
                    #     cv2.line(new_angles, point, nearest_point, val, 4)

                    # val = np.arctan(700/(dist_+2))*0.9

                    # if dist_ <= 5:
                    #     k = dist_/10
                    #     val = np.abs(angle*(1-k) * 0.95)

                    #     # val = angle*(1-k) * 0.95
                    # else:
                    #     k = dist_/5
                    # val = np.abs(angle*(1-k) * 0.95)
                    # if k < 1:
                    #     val = angle*(1-k) * 0.95
                        # val = angle/k * 0.95

                    # else:
                        # val = angle/k * 0.95


                    # ___ added - плохо видно широкую границу 9-10 пикселей

                    # if dist_ > 6:
                        # dist_ = 6
                    # k = (dist_-2)/10
                    # val = angle * (1 - k)
                    # val = angle * (1 - k)

                    # another add
                    # val = 0.9 * angle

                    # added second - пытаюсь улучшить видимость широкой границы
                    # if dist_>=8:
                    #     val = val * 1.1
                    # val = angle * 0.9
                    # print(dist_, angle, k, val)

                    # y_mean = val
                    # y_0 = val/2
                    # y_n = val/2
                    
                    # y_0 = val/6
                    # y_n = val/6
                    # x_plot = np.arange(0, len(discrete_line)+2)
                    # x_plot = np.arange(0, len(discrete_line))

                    # x_plot = np.arange(0, len(discrete_line)+1)

                    # coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                    # coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2 - 1], x_plot[-1]], [y_0, y_mean, y_n])

                    # coefs = curve_fit(parabola, [x_plot[0], x_plot[1], x_plot[len(x_plot)//2 - 1], x_plot[-2], x_plot[-1]], [0, y_0, y_mean, y_n, 0])


                    # a,b,c = coefs[0]
                    # y_parabola = parabola(x_plot, a, b, c)

                    # if dist_ > 10:
                    #     thickness = 5
                    # else:
                    #     thickness = 3
                    
                    # # discrete_line_ = prev + discrete_line + next
                    # discrete_line_ = discrete_line

                    # # discrete_line_ = prev + discrete_line

                    # draw_gradient_line(new_angles, point, discrete_line_, y_parabola, thickness=thickness)

    # print(np.unique(width_img == 3))
    # print(np.unique(width_img))
    mask_width = width_img == 3

    if len(np.unique(mask_width))>1:
    # print(np.unique(new_angles))
        val3 = np.max(new_angles[mask_width])
        new_angles[width_img == 2] = val3*1.001
        new_angles[width_img == 1] = val3*1.002
    else:
        new_angles[width_img == 2] = angles_img[width_img == 2] * 0.9
        new_angles[width_img == 1] = angles_img[width_img == 1] * 0.9

    mask = img != 128 

    width_img[mask] = 0
    angles_img[mask] = 0

    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole

    # mask2 = width_img < 4
    new_angles_copy = new_angles.copy()
    # new_angles = cv2.GaussianBlur(new_angles, (7, 7), 0)
    # new_angles[mask2] = new_angles_copy[mask2]
    new_angles[mask] = 0
    # angles_img = cv2.GaussianBlur(angles_img, (7, 7), 0)
    # angles_img = cv2.GaussianBlur(angles_img, (3, 3), 0)

    # color_map = cv2.GaussianBlur(color_map, (7, 7), 0)

    return width_img, angles_img, new_angles, color_map
    # return width_img
