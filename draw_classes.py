import cv2
import numpy as np
# import skimage as sk
# import matplotlib.pyplot as plt


class Figure():
    def __init__(self, figure_class):
        self.name = figure_class['FigureClass']
        self.margin = figure_class['figure_params']['margin']
        self.gap = figure_class['figure_params']['gap'] 
        self.corner_radius = figure_class['figure_params']['corner_radius']
        self.h = figure_class['figure_params']['h']
        self.w = figure_class['figure_params']['w']
        self.radius = figure_class['figure_params']['radius']
        self.color = figure_class['figure_params']['color']
        self.def_angle = figure_class['figure_params']['default angle']

        if 'Circle' in self.name:
            self.h = 2 * self.radius
            self.w = 2 * self.radius
        
        self.border_width = (700 / np.tan(np.radians(self.def_angle))) // 12
        self.border_color = 128
                 


    def draw_figure(self):
        if 'Circle' in self.name:
            # img = np.zeros((2*self.radius + 2*self.margin, 2*self.radius + 2*self.margin), dtype=np.float32)
            img = cv2.circle(img, (img.shape[0]//2, img.shape[1]//2), self.radius, 255, -1)
            img = cv2.circle(img, (img.shape[0]//2, img.shape[1]//2), self.radius +  self.border_width, 255, 1)
        else:
            img = np.zeros_like((2*np.max(self.h, self.w) + 2*self.margin, 2*np.max(self.h, self.w) + 2*self.margin), dtype=np.float32)
        
        return img
    
    def draw(self, img, position):
        if 'Circle' in self.name:
            img = cv2.circle(img, position, self.radius, self.color, -1)
        else:
            img = np.zeros_like((2*np.max(self.h, self.w) + 2*self.margin, 2*np.max(self.h, self.w) + 2*self.margin), dtype=np.float32)
        
        return img
    


class Frame():
    def __init__(self, img, edge_figure, linear_figure, num_linear_x, num_linear_y):
        self.frame_y = img.shape[0]
        self.frame_x = img.shape[1]

        self.num_linear_x = num_linear_x
        self.num_linear_y = num_linear_y
        
        self.num_edges = 4
        self.edge_figure = edge_figure
        self.linear_figure = linear_figure

        self.top_left = (edge_figure.margin + edge_figure.h // 2, edge_figure.margin + edge_figure.w // 2)
        self.bottom_right = (self.frame_x - edge_figure.margin - edge_figure.h // 2, self.frame_y - edge_figure.margin - edge_figure.w // 2)

        self.top_right = (self.frame_x - edge_figure.margin - edge_figure.h // 2, edge_figure.margin + edge_figure.w // 2)
        self.bottom_left = (edge_figure.margin + edge_figure.h // 2, self.frame_y - edge_figure.margin - edge_figure.w // 2)


        self.hor_size = (self.frame_x - edge_figure.w - edge_figure.margin) - (0+edge_figure.w + edge_figure.margin)
        self.vert_size = (self.frame_y - edge_figure.h - edge_figure.margin) - (0+edge_figure.h + edge_figure.margin)
        self.start_point = (edge_figure.h + edge_figure.margin, edge_figure.w + edge_figure.margin)

        self.positions = [self.top_left, self.bottom_right, self.top_right, self.bottom_left] # hor, vert

        self.gap_hor = (((((self.frame_x - edge_figure.w - edge_figure.margin) - (0+edge_figure.w + edge_figure.margin))\
                          )//self.num_linear_x) - linear_figure.w)

        self.gap_vert = (((((self.frame_y - edge_figure.h - edge_figure.margin) - (0+edge_figure.h + edge_figure.margin))\
                          )//self.num_linear_y) - linear_figure.h)
        

        self.gap_from_edge_figure_hor = self.edge_figure.h//2 + self.gap_hor
        self.gap_from_edge_figure_vert = self.edge_figure.w//2 + self.gap_vert

        
        self.linear_positions = [(self.top_left[0] + self.gap_from_edge_figure_hor, self.top_left[0]),\
                                 (self.bottom_left[0] + self.gap_from_edge_figure_hor, self.bottom_left[1]),\
                                 (self.top_left[0], self.top_left[0] + self.gap_from_edge_figure_vert),\
                                    (self.top_right[0], self.top_right[1] + self.gap_from_edge_figure_vert)
                                    ]
    
    def draw_frame(self, img):
        for position in self.positions:
            img = self.edge_figure.draw(img, position)

        for position in self.linear_positions[0:2]:
            start_hor = position[0]
            start_vert = position[1]  
            for i in range(0, self.num_linear_x):
                hor = start_hor + i * (self.linear_figure.w + self.gap_hor + (self.linear_figure.w//2))
                vert = start_vert
                img = self.linear_figure.draw(img, (hor, vert))

        for position in self.linear_positions[2:]:
            start_hor = position[0]
            start_vert = position[1]  
            for i in range(0, self.num_linear_y):
                hor = start_hor 
                vert = start_vert + i * (self.linear_figure.h + self.gap_vert + (self.linear_figure.h//2))
                img = self.linear_figure.draw(img, (hor, vert))

    
    def get_frame_size(self):
        frame_size = (self.vert_size, self.hor_size)
        start_point = self.start_point
        return frame_size, start_point


class Center():
    def __init__(self, start_point, size, figures, nums):
        self.start_point = start_point
        self.size = size
        self.figures = figures
        self.nums_figures = nums

        self.nums_at_all  = np.sum(nums)

        if self.nums_at_all % 3 == 0 and self.nums_at_all % 2:
            self.size_s_hor = self.size[1] // 3
            self.size_s_vert = self.size[0] // 2
            self.num_hor = 3
            self.num_vert = 2

        elif self.nums_at_all == 2:
            self.size_s_hor = self.size[1] // 2
            self.size_s_vert = self.size[0]
            self.num_hor = 2
            self.num_vert = 1
            
        elif self.nums_at_all % 2 == 0 and self.nums_at_all % 3 != 0:
            self.size_s_hor = self.size[1] // 2
            self.size_s_vert = self.size[0] // 2
            self.num_hor = 2
            self.num_vert = 2

        elif self.nums_at_all % 4 == 0 and self.nums_at_all % 3 == 0:
            self.size_s_hor = self.size[1] // 4
            self.size_s_vert = self.size[0] // 3
            self.num_hor = 4
            self.num_vert = 3

        elif self.nums_at_all % 4 == 0 and self.nums_at_all % 5 == 0:
            self.size_s_hor = self.size[1] // 5
            self.size_s_vert = self.size[0] // 4
            self.num_hor = 5
            self.num_vert = 4

        elif self.nums_at_all % 4 == 0 and self.nums_at_all // 4 > 1:
            self.size_s_vert = self.size[0] // 4
            self.size_s_hor = self.size[1] // self.size_s_vert
            self.num_hor = self.size_s_vert
            self.num_vert = 4

        elif self.nums_at_all == 1:
            self.size_s_hor = self.size[1]
            self.size_s_vert = self.size[0]
            self.num_hor = 1
            self.num_vert = 1

    def draw_center_figure(self, img):
        start_hor = self.start_point[1]
        start_vert = self.start_point[0]
        for hor_figures in range(self.num_hor):
            for i in self.nums_figures:
                for figure in self.figures:
                    center_hor = (start_hor + start_hor + self.size_s_hor) // 2
                    center_vert = (start_vert + start_vert + self.size_s_vert) // 2
                    figure.draw(img, (center_hor, center_vert))
                    start_hor = start_hor + self.size_s_hor
                self.nums_figures[self.nums_figures.index(i)] -= 1 

        start_hor = self.start_point[1]
        start_vert = self.start_point[0] + self.size_s_vert
        for hor_figures in range(self.num_hor):
            for i in self.nums_figures:
                if i != 0:
                    for figure in self.figures:
                        center_hor = (start_hor + start_hor + self.size_s_hor) // 2
                        center_vert = (start_vert + start_vert + self.size_s_vert) // 2
                        figure.draw(img, (center_hor, center_vert))
                        start_hor = start_hor + self.size_s_hor
                    self.nums_figures[self.nums_figures.index(i)] -= 1
