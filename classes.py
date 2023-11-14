
class Frame():
    def __init__(self, img, edge_figure, linear_figure, num_linear_x, num_linear_y):
        global count_figures
        global all_positions_dict
        global all_figures_dict
        count_figures = count_figures
        all_positions_dict = all_positions_dict
        all_figures_dict = all_figures_dict

        marg_hor = np.arange(20, 80, 10)
        marg_vert = np.arange(50, 110, 10)
        marg_ind = random.randint(0, len(marg_hor))
        self.frame_margin_hor = 20
        self.frame_margin_vert = 50

        self.lin_name = linear_figure.name
        self.edge_name = edge_figure.name
        num_bad = random.randint(0, 20)


        # self.frame_y = img.shape[0] # высота изобр
        # self.frame_x = img.shape[1] # ширина изобр

        # ИСПРАВИТЬ должно вычетаться два марджина
        self.frame_y = img.shape[0] - 2*self.frame_margin_vert # высота изобр
        self.frame_x = img.shape[1] - 2*self.frame_margin_hor # ширина изобр

        self.num_linear_x = num_linear_x # колво фигур в массиве горизонтальном
        self.num_linear_y = num_linear_y    # колво фигур в массиве вертикальном
        
        self.num_edges = 4
        self.edge_figure = edge_figure # фигура в углах рамки - объект класса Figure
        self.linear_figure = linear_figure  # фигура в массивах - объект класса Figure

        # координаты угловых фигур

        self.edge_positions, count_figures, all_positions_dict, all_figures_dict =  \
                                                                    self.calculate_edge_positions(edge_figure, count_figures, \
                                                                    all_positions_dict, all_figures_dict)
        print(self.edge_positions)
        
        self.hor_size = (self.frame_x - edge_figure.w - edge_figure.margin) - \
            (0+edge_figure.w + edge_figure.margin)
        self.vert_size = (self.frame_y - edge_figure.h - edge_figure.margin) - \
            (0+edge_figure.h + edge_figure.margin)
        self.start_point = (edge_figure.h + edge_figure.margin, \
                            edge_figure.w + edge_figure.margin)

        self.gap_hor = (((((self.frame_x - edge_figure.w) - \
                           (0+edge_figure.w)))//self.num_linear_x) - linear_figure.w)

        self.gap_vert = (((((self.frame_y - edge_figure.h) - \
                            (0+edge_figure.h)))//self.num_linear_y) - linear_figure.h)
        

        self.gap_from_edge_figure_hor = self.edge_figure.h//2 + self.linear_figure.h//2 + self.linear_figure.gap//2
        self.gap_from_edge_figure_vert = self.edge_figure.w//2 + self.linear_figure.w//2 + self.linear_figure.gap//2


        # позиции всех линейных фигур: вертикаль и горизонталь
        self.all_lin_positions = []
        self.linear_top_pos, self.all_lin_positions, count_figures, all_positions_dict, all_figures_dict = \
            self.calculate_lin_positions(linear_figure,self.edge_positions[0], self.edge_positions[2], [], \
                                self.all_lin_positions, self.num_linear_x, 'hor', count_figures, all_positions_dict, all_figures_dict)
        
        self.linear_bottom_pos, self.all_lin_positions, count_figures, all_positions_dict, all_figures_dict = \
            self.calculate_lin_positions(linear_figure,self.edge_positions[3], self.edge_positions[2], [], \
                                self.all_lin_positions, self.num_linear_x, 'hor', count_figures, all_positions_dict,all_figures_dict)

        self.linear_left_pos, self.all_lin_positions, count_figures, all_positions_dict, all_figures_dict = \
            self.calculate_lin_positions(linear_figure,self.edge_positions[0], self.edge_positions[3], [], \
                                self.all_lin_positions, self.num_linear_y, 'vert', count_figures, all_positions_dict, all_figures_dict)
        
        self.linear_right_pos, self.all_lin_positions, count_figures, all_positions_dict, all_figures_dict = \
            self.calculate_lin_positions(linear_figure,self.edge_positions[2], self.edge_positions[1], [], \
                                self.all_lin_positions, self.num_linear_y, 'vert', count_figures, all_positions_dict, all_figures_dict)


        self.positions = self.edge_positions + self.all_lin_positions
        self.bad_ind = random.sample(range(len(all_positions_dict)), num_bad)

        self.bad_position = [all_positions_dict[ind] for ind in self.bad_ind]
        self.bad_type = [all_figures_dict[ind] for ind in self.bad_ind]
        
        
    def calculate_edge_positions(self, edge_figure, count_figures, all_positions_dict, all_figures_dict):
        self.top_left = (self.frame_margin_hor + edge_figure.h // 2, self.frame_margin_vert + edge_figure.w // 2) # координата центра верхней левой
        self.bottom_right = (self.frame_x +  self.frame_margin_hor - edge_figure.h // 2, \
                             self.frame_y + self.frame_margin_vert - edge_figure.w // 2)    # координата центра правой нижней

        self.top_right = (self.frame_x +  self.frame_margin_hor - edge_figure.h // 2, \
                          self.frame_margin_vert + edge_figure.w // 2)  # координата центра правой верхней
        self.bottom_left = (self.frame_margin_hor + edge_figure.h // 2, \
                            self.frame_y + self.frame_margin_vert - edge_figure.w // 2) # координата центра левой нижней


        self.edge_positions = [self.top_left, self.bottom_right, self.top_right, self.bottom_left] # позиции угловых фигур

        all_positions_dict[count_figures] = self.top_left
        all_positions_dict[count_figures+1] = self.bottom_right
        all_positions_dict[count_figures+2] = self.top_right
        all_positions_dict[count_figures+3] = self.bottom_left

        all_figures_dict[count_figures] = edge_figure.name
        all_figures_dict[count_figures+1] = edge_figure.name
        all_figures_dict[count_figures+2] = edge_figure.name
        all_figures_dict[count_figures+3] = edge_figure.name

        count_figures = count_figures + 3

        return self.edge_positions, count_figures, all_positions_dict, all_figures_dict


    def calculate_lin_positions(self, linear_figure, start_pos, end_pos, lin_positions, all_lin_pos, num, \
                                direction, count_figures, all_positions_dict, all_figures_dict):
        if direction == 'hor':
            start_hor = start_pos[0] + self.gap_from_edge_figure_hor
            end_hor = end_pos[0] - self.gap_from_edge_figure_hor
            lin_pos_hor = np.linspace(start_hor, end_hor, num)
            start_vert = start_pos[1]
            for i in range(num):
                pos_hor = int(lin_pos_hor[i])
                pos_vert = start_vert
                lin_positions.append((pos_hor, pos_vert))
                all_lin_pos.append((pos_hor, pos_vert))
                count_figures += 1
                all_positions_dict[count_figures] = (pos_hor, pos_vert)
                all_figures_dict[count_figures] = linear_figure.name

        if direction == 'vert':
            start_hor = start_pos[0]
            start_vert = start_pos[1] + self.gap_from_edge_figure_vert
            end_vert = end_pos[1] - self.gap_from_edge_figure_vert
            lin_pos_vert = np.linspace(start_vert, end_vert, num)
            for i in range(num):
                pos_hor = start_hor
                pos_vert = int(lin_pos_vert[i])
                lin_positions.append((pos_hor, pos_vert))
                all_lin_pos.append((pos_hor, pos_vert))
                count_figures += 1
                all_positions_dict[count_figures] = (pos_hor, pos_vert)
                all_figures_dict[count_figures] = linear_figure.name

        return lin_positions, all_lin_pos, count_figures, all_positions_dict, all_figures_dict


    def draw_frame(self, img):
        for i, position in enumerate(self.positions):
            if all_figures_dict[i] == self.lin_name:
                if i in self.bad_ind:
                    img = self.linear_figure.draw(img, position, True)
                else:
                    img = self.linear_figure.draw(img, position, False)

            if all_figures_dict[i] == self.edge_name:
                if i in self.bad_ind:
                    img = self.edge_figure.draw(img, position, True)
                else:
                    img = self.edge_figure.draw(img, position, False)
    
    def get_frame_size(self):
        frame_size = (self.vert_size, self.hor_size)
        start_point = self.start_point
        return frame_size, start_point
