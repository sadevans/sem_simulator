import cv2
import numpy as np


class FigureClass():
    def __init__(self, hole_contour):
        self.hole_contour = hole_contour.copy()
        self.border_width = 0
        self.border_contour = hole_contour.copy()

    def getContour(self):
        return self.hole_contour
    
    def getBorderContour(self):
        return self.border_contour
    
    def init_border_contour(self, img, object, width = None):
        """ Метод, ищутся все точки контура границы и пишутся в соответсвующее поле"""
        if width is None:
            width = object.border_width
        temp = np.zeros_like(img)                     # создание временной картинки
        cv2.drawContours(temp, [object.contour], 0, 1, 2*width) # рисование контура с первоначальным offset
        c, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # детектирование контура
        object.border_contour = c[0].reshape(-1, 2).copy()

    
    def try_render(self, img, objects, object, i, width):
        """ Метод, в котором пытаешься отрендерить контур с границей и проверяешь, 
        не пересекается ли он с другими границами или объектами"""
        object.border_width = width
        self.init_border_contour(img, object)
        
        obj_num = None
        flag_render = True
        for k in range(len(objects)):
            set_cur_cont = set(map(tuple, object.border_contour))

            if k != i:
                set_another_cont = set(map(tuple, objects[k].border_contour))

                if set_cur_cont & set_another_cont:
                    obj_num = k
                    flag_render = False
                    break
        
        return obj_num, flag_render
    
        
class Image():
    def __init__(self, mask: np.array):
        self.mask = mask
        self.color_map = np.zeros_like(mask)
        self.angles_map = np.zeros_like(mask)
        self.width_map = np.zeros_like(mask)

        self.signal = np.zeros_like(mask)
        self.noisy = np.zeros_like(mask)


        self.objects = []

    
    def detect_cont(self, img):
        cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont


    def init_contours(self):
        mask_hole = cv2.inRange(self.mask, 255, 255)
        mask_border = cv2.inRange(self.mask, 128, 128)

        c_int_ = self.detect_cont(mask_hole)
        c_ext_ = self.detect_cont(mask_border)

        for i in range(len(c_ext_)):
            for j in range(len(c_int_)):
                temp_bord = np.zeros_like(self.mask)
                cv2.drawContours(temp_bord, [c_ext_[i]], -1, 128, -1)
                temp_hole = np.zeros_like(self.mask)
                cv2.drawContours(temp_hole, [c_int_[j]], -1, 255, -1)
                set_border = set(map(tuple, np.argwhere(temp_bord==128)))
                set_hole = set(map(tuple, np.argwhere(temp_hole==255)))
                if set_hole.issubset(set_border):
                    obj = FigureClass(c_int_[j].reshape(-1, 2)) # создаем объекты класса FigureClass
                    obj.border_contour = c_ext_[i].reshape(-1, 2).copy()
                    self.objects.append(obj)