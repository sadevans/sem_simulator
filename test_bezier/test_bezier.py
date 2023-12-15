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
print(device)

from all import *
from transform import *
from transform_radius import *

from draw import *


from skimage.draw import line

def bezier(line, t, prev, color_hole, color_back):
    if prev == 0.0:
        point1 = (0, color_back)
        point4 = (len(line), color_hole)
        eq = lambda point: ((point - point1[0])/(point4[0] - point1[0]))*(point4[1] - point1[1]) + point1[1]
        x3 = random.randint(0, len(line)-1)
        y3 = random.uniform(color_hole, eq(x3))
        # print(x3, y3)
        point3 = (0, color_hole)
        # point3 = (x3, y3)
        x2 = random.randint(1, len(line))
        y2 = random.uniform(eq(x2)+1,color_back)
        # print(x2, y2)
        # point2 = (x2, y2)
        point2 = (len(line), color_back)
    if prev == 255.0:
        # print('white')

        point1 = (0, color_hole)
        point4 = (len(line), color_back)
        eq = lambda point: ((point - point1[0])/(point4[0] - point1[0]))*(point4[1] - point1[1]) + point1[1]
        x3 = random.randint(0, len(line)-1)
        y3 = random.uniform(eq(x3), color_back+1)
        # print(x3, y3)
        point3 = (0, color_back)
        # point3 = (x3, y3)
        x2 = random.randint(1, len(line))
        y2 = random.uniform(color_hole, eq(x2))
        # print(x2, y2)
        # point2 = (x2, y2)
        point2 = (len(line), color_hole)

    # point2 = (len(line), 97.5)
    # point2 = (x2, y2)
    x = point1[0]*(1-t)**3 + point2[0]*3*t*(1-t)**2 + point3[0]*3*t**2*(1-t) + point4[0]*t**3
    vals = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3
    return x, vals


def transform1(img, ext, int):
    color_back = 110
    color_hole = 85

    width_img = np.zeros_like(img, dtype=np.float32)
    angles_img = np.zeros_like(img, dtype=np.float32)
    new_angles = np.zeros_like(img, dtype=np.float32)
    color_map = np.zeros_like(img, dtype=np.float32)
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole

    fig, ax = plt.subplots(1,1,figsize=(10, 10))
    cp = np.zeros_like(circle2)
    # cp = cv2.fillPoly(cp, [int], 2)
    # cp = cv2.fillPoly(cp, [cont_int], 2)


    count = 0
    for cont_ext, cont_int in zip(ext, int):
        # cp = cv2.fillConvexPoly(cp, cont_int, 0)
        # cp = cv2.fillConvexPoly(cp, cont_ext, 0)
        cp = cv2.fillPoly(cp, [cont_ext], 1)

        cp = cv2.fillPoly(cp, [cont_int], 2)
        # cp = cv2.fillPoly(cp, [cont_ext], 1)
        # cont_mask, _ = cv2.findContours(cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # mask_cont = np.zeros_like(circle2)
        # cv2.drawContours(mask_cont, cont_mask,  -1, 1, 0)
        # cp = np.zeros_like(circle2, dtype=np.float32)
        # cp = cv2.fillPoly(cp, [cont_ext], 1)
        # cont_mask, _ = cv2.findContours(cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # mask_cont = np.zeros_like(circle2)
        # cv2.drawContours(mask_cont, cont_mask,  -1, 1, 0)

        ax.imshow(cp)

        for point in cont_ext:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_int)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_int[index] 
                    prev = [compute_previous_pixel(point, nearest_point)]
                    
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

                    # if (point[0] == nearest_point[0]) or (point[1] == nearest_point[1]):
                        # dist_ += 1
                        # print(dist_)
                        # next = [compute_next_pixel(point, nearest_point)]
                        # discrete_line = discrete_line + next

                        # discrete_line = list(zip(*line(*prev[0], *nearest_point))) # find all pixels from the line

                    # discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line

                # dist_ += 1
                
                # if dist_ == 0:
                #     dist_ = 1

                # discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
                # discrete_line = list(zip(*line(*prev[0], *nearest_point))) # find all pixels from the line

                # discrete_line_x = np.array(list(zip(*discrete_line))[0])
                # if len(discrete_line) > 1:
                #     mid_point = [(point[0] + nearest_point[0])//2, (point[1] + nearest_point[1])//2]
                #     prev = [compute_previous_pixel(point, nearest_point)]
                #     next = [compute_next_pixel(point, nearest_point)]
                # print(point)
                # print('color point = {:.8f}, color prev = {:.8f}, len line = {:.8f}'.format(img[point[1], point[0]], img[prev[0][1], prev[0][0]], len(discrete_line)))
                # if img[prev[0][1], prev[0][0]] == 128.0:
                #     # print('color point = {:.8f}, color prev = {:.8f}, len line = {:.8f}'.format(img[point[1], point[0]], img[prev[0][1], prev[0][0]], len(discrete_line)))
                #     dist += 1
                #     # count += 1
                #     prev = [compute_previous_pixel(prev[0], nearest_point)]
                    

                # ax.scatter(point[0], point[1], color='red', s=2, alpha=0.5)
                # ax.scatter(prev[0][0], prev[0][1], color='yellow', s=0.5, alpha=0.5)
                # ax.scatter(next[0][0], next[0][1], color='green', s=0.5, alpha=0.5)
                # ax.scatter(nearest_point[0], nearest_point[1], color='blue', s=2, alpha=0.5)
                # if dist_ == 7:
                # ax.plot([point[0], nearest_point[0]], [point[1], nearest_point[1]], color='orange', linewidth=1)
                # ax.scatter(mid_point[0], mid_point[1], color='hotpink', s=1, alpha=0.5)

                # if img[prev[0][1], prev[0][0]] == 128.0:
                    # ax.scatter(point[0], point[1], color='red', s=2, alpha=0.5)
                    # ax.scatter(prev[0][0], prev[0][1], color='yellow', s=0.5, alpha=0.5)
                    # ax.scatter(next[0][0], next[0][1], color='green', s=0.5, alpha=0.5)
                    # ax.scatter(nearest_point[0], nearest_point[1], color='blue', s=2, alpha=0.5)

                #     print('!!!!!!!!!!!!!!!')
                #     # dist_ += 1
                #     # prev = [compute_previous_pixel(prev[0], nearest_point)]
                #     # discrete_line = prev + discrete_line
                #     dist += 1
                #     next = [compute_next_pixel(point, next[0])]
                #     discrete_line = discrete_line + next

                new_line = np.zeros(dist_*12, dtype=np.float32)
                # x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), img[prev[0][1], prev[0][0]], color_hole, color_back)
                # x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, color_hole, color_back)
                x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, 100, 700)
                x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, color_hole, color_back)
                new_y = y[::12]
                reshaped_y  = np.array(y).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                # averages_y = np.mean(reshaped_y, axis=1)
                averages_y = np.max(reshaped_y, axis=1)

                reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                # averages_y = np.mean(reshaped_y, axis=1)
                # averages_colors = np.max(reshaped_colors, axis=1)


                # angles = np.arctan(np.abs(np.gradient(y, x)))
                angles = np.arctan(np.abs(np.gradient(y)))

                new_angl = angles[::12]

                reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                # averages_angls = np.mean(reshaped_angls, axis=1)

                averages_angls = np.max(reshaped_angls, axis=1)
                max_indices = np.argmax(reshaped_angls, axis=1)
                # print(max_indices)
                averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]

                # averages_angls[averages_angls < 0.1] = 0.1

                # if dist_ <= 3:
                #     averages_angls = angles[::12]
                    # averages_angls = np.max(reshaped_angls, axis=1)
                # averages_angls[averages_angls < 0.1] = 0.1

                # print(averages_angls)

                if len(averages_angls) != len(discrete_line):
                    print('dist = {:.2f}, len angls = {:.2f}, len line = {:.2f}'.format(dist_, len(averages_angls), len(discrete_line)))
                if dist_ > 10:
                    draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=3)
                    draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=4)

                else:
                    draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=4)
                    draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)
                     
                # draw_gradient_line(color_map, point, discrete_line, new_y, thickness=3)
                # draw_gradient_line(new_angles, point, discrete_line, new_angl, thickness=1)
                # draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)
            
                cv2.line(width_img, point, nearest_point, dist_, 3)


    for cont_ext, cont_int in zip(ext, int):
        for point in cont_int:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_ext)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_ext[index]
                    # prev = [compute_previous_pixel(point, nearest_point)]
                    next = [compute_next_pixel(point, nearest_point)]
                    discrete_line = list(zip(*line(*point, *next[0]))) # find all pixels from the line
                    dist_ = len(discrete_line)
                    # if nearest_point[0] != point[0] or nearest_point[1] != point[1]:
                    #     dist_ -= 1
                    # else:
                        # dist_ = np.sqrt((point[0] - nearest_point[0])**2 + (point[1] - nearest_point[1])**2).astype(np.int32)
                    # dist_ = np.sqrt((prev[0][0] - nearest_point[0])**2 + (prev[0][1] - nearest_point[1])**2).astype(np.int32)
                    # dist_ = np.sqrt((next[0][0] - nearest_point[0])**2 + (next[0][1] - nearest_point[1])**2).astype(np.int32)
                    
                    # dist_ = dist_.item()

                # dist_ += 1
                if dist_ == 0:
                    dist_ = 1

                # print(dist_)
                # if dist_< 19:
                #     print('!!!!!!!!!!')

                # if dist_ > 10:
                #     dist_ = 10
                # discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
            
                # discrete_line_x = np.array(list(zip(*discrete_line))[0])
                # if len(discrete_line) > 1:
                #     mid_point = [(point[0] + nearest_point[0])//2, (point[1] + nearest_point[1])//2]
                #     prev = [compute_previous_pixel(point, nearest_point)]
                #     next = [compute_next_pixel(point, nearest_point)]

                # if img[prev[0][1], prev[0][0]] == 128.0:
                #     dist += 1
                #     discrete_line = prev + discrete_line

                #     prev = [compute_previous_pixel(prev[0], nearest_point)]

                # if img[prev[0][1], prev[0][0]] == 128.0:
                # if img[next[0][1], next[0][0]] == 128.0:
                #     print('!!!!!!!!!!!')
                    # dist += 1
                    # next = [compute_next_pixel(point, next[0])]
                    # discrete_line = discrete_line + next


                new_line = np.zeros(dist_*12, dtype=np.float32)
                # x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), img[prev[0][1], prev[0][0]], color_hole, color_back)
                # x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, color_hole, color_back)
                x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, 100, 700)
                x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, color_hole, color_back)

                new_y = y[::12]
                reshaped_y  = np.array(y).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                # averages_y = np.mean(reshaped_y, axis=1)
                averages_y = np.max(reshaped_y, axis=1)

                reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                # averages_y = np.mean(reshaped_y, axis=1)
                # averages_colors = np.max(reshaped_colors, axis=1)

                # angles = np.arctan(np.abs(np.gradient(y, x)))
                angles = np.arctan(np.abs(np.gradient(y)))

                new_angl = angles[::12]

                reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                
                # averages_angls = np.mean(reshaped_angls, axis=1)
                averages_angls = np.max(reshaped_angls, axis=1)
                # averages_angls = reshaped_angls[np.arange(len(reshaped_angls)), max_indices]

                max_indices = np.argmax(reshaped_angls, axis=1)
                averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                if len(averages_angls) != len(discrete_line):
                    print('dist = {:.2f}, len angls = {:.2f}, len line = {:.2f}'.format(dist_, len(averages_angls), len(discrete_line)))
                # draw_gradient_line(color_map, point, discrete_line, np.clip(averages_y, 85, 110), thickness=3)
                
                # if dist_ > 10:
                #     draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=0)
                #     draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=0)
                draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=1)

                if dist_ > 10:
                    if point[0] != nearest_point[0] and point[1] != nearest_point[1]:
                        draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=4)
                        draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)
                if dist_ <= 10:
                    draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=4)
                    draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)

                # # draw_gradient_line(new_angles, point, discrete_line, new_angl, thickness=1)
                # draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)
  
    # mask_width = width_img == 3
    mask = img != 128 
    width_img[mask] = 0
    # new_angles = cv2.GaussianBlur(new_angles, (5,5), 0)
    new_angles[mask] = 0
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole


    # color_map = cv2.GaussianBlur(color_map, (5, 5), 0)
    # print(max_indices, reshaped_colors, averages_colors)
    return width_img, color_map, new_angles


def formula_second(img, angles, color_map, k, file_name, save_dir):
    print(f'{save_dir}/{file_name}')
    signal = np.zeros_like(img, dtype=np.float32)
    alpha_bord = angles[img == 128]
    # alpha_bord[alpha_bord==alpha_bord.min()] = np.radians(1)
    alpha_back = angles[img == 0]
    print(np.unique(alpha_back))
    alpha_hole = angles[img == 255]
    # print(np.unique(alpha_hole), np.unique(alpha_back))
    # k = k * 
    signal[img == 0] = (k*(1/(np.abs(np.cos(np.radians(alpha_back + 1)))**(0.87)) - 1) + 1) * color_map[img==0]

    signal[img == 128] = (k * (1/(np.abs(np.cos(np.radians(90)-(np.radians(180 - 90) - alpha_bord)))**(0.87)) - 1) + 1) *color_map[img==128]
    signal[img == 255] = (k * (1 / (np.abs(np.cos(np.radians(alpha_hole + 1)))**(1.1)) - 1) + 1) * color_map[img==255]
    signal = np.clip(signal, 0, 255)
    # signal = cv2.GaussianBlur(signal, (11,11), 0)
    signal = cv2.GaussianBlur(signal, (9,9), 0)
    cv2.imwrite(f'{save_dir}/{file_name[:-4]}.png', signal.astype(np.uint8))
    return signal


if __name__ == '__main__':
    w = 4
    # circle2 = simulate_circle(w)
    # circle2 = simulate_circles(radius=50)
    circle2 = simulate_squares(radius=50)
    # circle2 = simulate_squares_1120(radius=50)
    # circle2 = simulate_circles_1120(radius=50)



    # cv2.imwrite(f'circles.png', circle2.astype(np.uint8))
    # cv2.imwrite(f'squares.png', circle2.astype(np.uint8))
    # cv2.imwrite(f'squares_1120.png', circle2.astype(np.uint8))
    # cv2.imwrite(f'circles_1120.png', circle2.astype(np.uint8))

    ext, int, circle2 = detect_contours(circle2)
    width, color_map, new_angles = transform1(circle2, ext, int) # bezier
    # width, angles_img, new_angles, color_map = transform(circle2, ext, int) # parabola
    # width, angles_img, new_angles, color_map = transform_radius(circle2, ext, int) # parabola radius
    # k=0.125
    # k = 0.125
    k = 0.5
    signal_second = formula_second(circle2, new_angles, color_map, k, 'signal_circles.png', '.')
    # cv2.imwrite(f'signal_circle1_{w}_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_squares_bezier_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_circles1120_bezier_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_circles_bezier_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_squares1120_bezier_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_squares_parabola_radius_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_squares1120_parabola_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_squares_parabola_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_squares_parabola_radius_{k}.png', signal_second.astype(np.uint8))


    # cv2.imwrite(f'signal_squares1120_parabola_radius_{k}.png', signal_second.astype(np.uint8))

    # cv2.imwrite(f'signal_circles1120_parabola_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal_circles_parabola_{k}.png', signal_second.astype(np.uint8))

    # cv2.imwrite(f'signal_test_bezier.png', signal_second.astype(np.uint8))



    a = 50
    b = 10

    m = Gamma(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))
    sample = m.sample()
    sample.item()

    clean = torch.Tensor(signal_second)
    # clean = torch.Tensor(signal_first)

    noisy = clean +  m.sample()* torch.randn(clean.shape)
    # res = clean + np.clip(m.sample()* torch.randn(clean.shape), 0 , 255) # make salt noise
    # res = np.clip(clean + m.sample()* torch.randn(clean.shape), 0 , 255)
    # res = np.clip(clean,0, 255) + m.sample()* torch.randn(clean.shape)
    res = clean + m.sample()* torch.randn(clean.shape)

    # cv2.imwrite(f'signal_squares_bezier_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_circles_bezier_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_circles1120_bezier_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))



    # cv2.imwrite(f'signal_squares1120_bezier_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_squares1120_parabola_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_squares_parabola_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_squares_parabola_radius_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))


    # cv2.imwrite(f'signal_squares1120_parabola_radius_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_circles1120_parabola_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))
    # cv2.imwrite(f'signal_circles_parabola_{k}_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))






    # cv2.imwrite(f'signal_circles_parabola_radius_{k}.png', signal_second.astype(np.uint8))
    # cv2.imwrite(f'signal1_circles_bezier_{k}.png', signal_second.astype(np.uint8))
    y = 100

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax[0,0].imshow(width)
    ax[0,0].set_title(f'width')

    ax[0,1].imshow(color_map)
    ax[0,1].set_title(f'color_map')

    ax[0,2].imshow(new_angles)
    ax[0,2].set_title(f'new angles')

    ax[1,1].plot(color_map[y, :])
    ax[1,1].grid()


    ax[1,0].plot(width[y, :])
    ax[1,0].grid()
    ax[1,0].set_title(f'width')
    ax[1,1].set_title(f'color_map')

    ax[1,2].plot(new_angles[y, :])
    ax[1,2].grid()

    ax[1,2].set_title(f'new_angles')
    plt.show()

    # y = 60
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].imshow(signal_second)
    ax[0].set_title(f'signal')

    ax[1].plot(signal_second[y, :])
    ax[1].set_title(f'signal')
    ax[1].grid()
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].imshow(res)
    ax[0].set_title(f'noisy')

    ax[1].plot(res[y, :])
    ax[1].set_title(f'noisy signal')
    ax[1].grid()
    plt.show()