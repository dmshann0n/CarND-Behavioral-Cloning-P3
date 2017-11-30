""" Manages entries from driving data """

import cv2

def load_image(path):
    return cv2.imread(path)

class DrivingEntry:
    """ Encapsulates data and access to imgs from the driving log """

    def __init__(
            self, 
            center_img_path, 
            left_img_path, 
            right_img_path, 
            steering_angle, 
            throttle, 
            brake, 
            speed,
            src):

        self.center_img_path = center_img_path
        self.left_img_path = left_img_path
        self.right_img_path = right_img_path
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.brake = brake
        self.speed = speed
        self.src = src

    def left_img(self):
        return load_image(self.left_img_path)

    def right_img(self):
        return load_image(self.right_img_path)

    def center_img(self):
        return load_image(self.center_img_path)