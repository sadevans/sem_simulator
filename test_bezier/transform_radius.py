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

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def transform_radius(circle2, ext, int):
    color_back = 110
    color_hole = 85

    width_img = circle2.copy()
    angles_img = circle2.copy()
    # new_angles = circle2.copy()
    new_angles = np.zeros_like(circle2, dtype=np.float32)
    height_img = circle2.copy()
    height_img[circle2 == 0] = color_back
    height_img[circle2 == 255] = color_hole

    width_img[circle2 == 0] = 0
    width_img[circle2 == 255] = 0
    # new_angles[circle2 == 255] = 0
    flag = True


    for cont_ext, cont_int in zip(ext, int):
        mask = np.zeros_like(circle2)
        mask = cv2.fillPoly(mask, [cont_int], 2)
        img = np.zeros_like(circle2)
        modes = ['internal', 'external']

        # for mode in modes:
        #     if mode == 'external':
        cont = cont_ext.copy()
                # other_cont = cont_int
            # else:
            #     cont = cont_int.copy()
        for point in cont:
            # if mode == 'external':
            #     if width_img[point[1], point[0]] != 128:
            #         continue
            
            radius = 1
            flag = True
            while flag:
                mask = np.zeros_like(circle2)
                mask = cv2.circle(mask, point, radius, 255, -1)
                mask = cv2.inRange(mask, 255,255)
                cont_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mask_cont = np.zeros_like(circle2)
                cv2.drawContours(mask_cont, cont_mask,  -1, 1, 0)
                nonzero = np.argwhere(mask_cont > 0)
                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                # if mode == 'internal':
                #     intersect = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_ext)])
                # else:
                intersect = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_int)])
                    # intersect
                if len(intersect) != 0:
                    flag=False
                else:
                    radius += 1

            dist = float('inf')
            for dot in intersect:
                dist_ = np.sqrt((point[0] - dot[0])**2 + (point[1] - dot[1])**2)
                if dist_ < dist:
                    dist = ((np.round(dist_, 0)).astype(np.int32)).item() + 1
                    nearest_point = dot
            
            discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
            # discrete_line_x = np.array(list(zip(*discrete_line))[0])
            
            mid_point = [(point[0] + nearest_point[0])//2, (point[1] + nearest_point[1])//2]
            prev = [compute_previous_pixel(point, nearest_point)]
            next = [compute_next_pixel(point, nearest_point)]


            angle = np.arctan(700 / (dist * 12))
            cv2.line(width_img, point, nearest_point, dist, 3)
            cv2.line(angles_img, point, nearest_point, angle, 2)

            if len(discrete_line) == 1:
                height_vals = [color_back - 2]
                discrete_line_ =discrete_line

            else:
                height_vals = [((k_ * ((color_back - color_hole) / dist)) + color_hole) for k_ in range(1, dist+2)]
                # discrete_line_ = discrete_line + next
                discrete_line_ = discrete_line

            draw_gradient_line(height_img, point, discrete_line_, height_vals, thickness=3)


            # calculate new angles
            if dist_ >= 2:
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
                    draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=1)

    for cont_ext, cont_int in zip(ext, int):
        mask = np.zeros_like(circle2)
        mask = cv2.fillPoly(mask, [cont_int], 2)
        img = np.zeros_like(circle2)
        modes = ['internal', 'external']

        # for mode in modes:
        #     if mode == 'external':
        cont = cont_int.copy()
                # other_cont = cont_int
            # else:
            #     cont = cont_int.copy()
        for point in cont:
            # if mode == 'external':
            #     if width_img[point[1], point[0]] != 128:
            #         continue
            
            radius = 1
            flag = True
            while flag:
                mask = np.zeros_like(circle2)
                mask = cv2.circle(mask, point, radius, 255, -1)
                mask = cv2.inRange(mask, 255,255)
                cont_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mask_cont = np.zeros_like(circle2)
                cv2.drawContours(mask_cont, cont_mask,  -1, 1, 0)
                nonzero = np.argwhere(mask_cont > 0)
                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                # if mode == 'internal':
                #     intersect = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_ext)])
                # else:
                intersect = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_ext)])

                    # intersect
                if len(intersect) != 0:
                    flag=False
                else:
                    radius += 1

            dist = float('inf')
            for dot in intersect:
                dist_ = np.sqrt((point[0] - dot[0])**2 + (point[1] - dot[1])**2)
                if dist_ < dist:
                    dist = ((np.round(dist_, 0)).astype(np.int32)).item() + 1
                    nearest_point = dot
            
            discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
            discrete_line_x = np.array(list(zip(*discrete_line))[0])
            
            mid_point = [(point[0] + nearest_point[0])//2, (point[1] + nearest_point[1])//2]
            prev = [compute_previous_pixel(point, nearest_point)]
            next = [compute_next_pixel(point, nearest_point)]


            angle = np.arctan(700 / (dist * 12))
            cv2.line(width_img, point, nearest_point, dist, 3)
            cv2.line(angles_img, point, nearest_point, angle, 2)

            if len(discrete_line) == 1:
                height_vals = [color_back - 2]
                discrete_line_ =discrete_line

            else:
                height_vals = [((k_ * ((color_back - color_hole) / dist)) + color_hole) for k_ in range(1, dist+2)]
                # discrete_line_ = discrete_line + next
                discrete_line_ = discrete_line


            # discrete_line_ =discrete_line + next
            if point[0] != nearest_point[0] and point[1] != nearest_point[1]:
                draw_gradient_line(height_img, point, discrete_line_, height_vals[-1::-1], thickness=3)

            else: draw_gradient_line(height_img, point, discrete_line_, height_vals, thickness=3)

            # if dist_ <= 10: draw_gradient_line(height_img, point, discrete_line_, height_vals, thickness=3)
            # calculate new angles
            # if dist_ >= 2 and dist_ <= 10:
            if dist_ >= 2:


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
                    draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=1)

    mask_width = width_img == 3
    img = circle2.copy()
    # if len(np.unique(mask_width))>1:
    # # print(np.unique(new_angles))
    #     val3 = np.max(new_angles[mask_width])
    #     new_angles[width_img == 2] = val3*1.001
    #     new_angles[width_img == 1] = val3*1.002
    # else:
    #     new_angles[width_img == 2] = angles_img[width_img == 2] * 0.9
    #     new_angles[width_img == 1] = angles_img[width_img == 1] * 0.9

    mask = img != 128 

    width_img[mask] = 0
    angles_img[mask] = 0

    height_img[img == 0] = color_back
    height_img[img == 255] = color_hole

    # mask2 = width_img < 4
    new_angles_copy = new_angles.copy()
    # new_angles = cv2.GaussianBlur(new_angles, (7, 7), 0)
    # new_angles[mask2] = new_angles_copy[mask2]
    new_angles[mask] = 0
    # height_img = cv2.GaussianBlur(height_img, (7, 7), 0)
    # angles_img = cv2.GaussianBlur(angles_img, (3, 3), 0)
    # angles_img = cv2.GaussianBlur(angles_img, (3, 3), 0)

    # color_map = cv2.GaussianBlur(color_map, (7, 7), 0)

    return width_img, angles_img, new_angles, height_img
    # return width_img
