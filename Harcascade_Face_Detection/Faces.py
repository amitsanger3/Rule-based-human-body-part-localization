import cv2

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import os, json
import random
import pandas as pd


class Faces(object):
    
    def __init__(self, face_cascade):
        self.face_cascade = face_cascade
        self.error = []
        
    def face_co_ordinates(self, gray):
        return self.face_cascade.detectMultiScale(gray, 1.2, 4)

    @staticmethod
    def draw_rectangle(image, co_ordinate):
        x,y,w,h = co_ordinate
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        return None

    @staticmethod
    def crop_face(image, co_ordinate):
        x,y,w,h = co_ordinate
        return image[y:y+h, x:x+w]

    @staticmethod
    def save_face(output_dir, image_name, cropped_img):
        name = os.path.join(output_dir, image_name)
        cv2.imwrite(name, cropped_img)








