import cv2

import numpy as np
import matplotlib.pyplot as plt

import os, json
import random, traceback

import time
from tqdm import tqdm

import shutil

import torch
from torch import nn
import torch.nn.functional as F
# from sklearn.model_selection import train_test_split

from rembg import remove


class FrameDetector(object):
    
    def __init__(self):
        pass

    @staticmethod
    def neck_coordinates(x,y,w,h):
        return int(x+(w/8)), (y+h), int(w-(w/4)), int(h*0.4)
    
    @staticmethod
    def deep_neck_coordinates(x1,y1,w,h,h1):
        return int(x1 - (w / 8)), y1 + h1, w, h
    
    @staticmethod
    def upper_half_torso_coordinates(x,y,w,h, h1):
        return int(x - (w/2)), (y+h+h1), 2*w, 2*h

    @staticmethod
    def lower_half_torso_coordinates(x,y,w,h, h_):
        return x, y+h, w, h_
    
    @staticmethod
    def left_arm_coordinates(b_x1, y2, x3, y3, h3):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        return  b_x1, y2, x3, y3 + h3 + h3

    @staticmethod
    def left_hand_coordinates(b_x1, x3, y3, h3, h):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        return b_x1, y3 + h3 + h3 - h, x3, y3 + h3 + h3

    @staticmethod
    def right_arm_coordinates(b_x2, y2, x3, y3, h3, w3):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        return x3 + w3, y2, b_x2, y3 + h3 + h3

    @staticmethod
    def right_hand_coordinates(b_x2, x3, y3, h3, h, w3):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        return x3 + w3, y3 + h3 + h3 - h, b_x2, y3 + h3 + h3
    
    @staticmethod
    def legs_coordinates(x3,y3,w3,h3, b_y2):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        x4, y4, x4_, y4_ = np.array([x3, y3 + h3, x3 + w3, b_y2])
        return x4, y4, x4_, y4_

    @staticmethod
    def foot_coordinates(x4,x4_, y4_,h):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        x5, y5, x5_, y5_ = np.array([x4, y4_ - h, x4_, y4_])
        return x5, y5, x5_, y5_

    @staticmethod
    def dress_fullness(b_x1, y2, h2, b_x2, b_y2):
        """
        THis will return coordinates in upper left corner and lower right corner.
        """
        x6, y6, x6_, y6_ = np.array([b_x1, y2 + h2, b_x2, b_y2])
        return x6, y6, x6_, y6_
    
    @staticmethod
    def colors():
        return (random.choice(range(0, 255)), random.choice(range(0, 255)), random.choice(range(0, 255)))

    @staticmethod
    def hex_to_rgb(hex_color):
        """Converts a hex color to RGB.

        Args:
          hex_color: A hex color string, e.g. '#FFFFFF'.

        Returns:
          A tuple of RGB values, e.g. (255, 255, 255).
        """

        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb

    def draw_rectangle(self, image, coordinates, color, text, bb_box=True):
        img_height,_,_ = image.shape
        color = self.hex_to_rgb(color)
        if bb_box:
            x,y,w,h = coordinates
            x1, y1, x2, y2 = x, y, x+w, y+h
        else:
            x1, y1, x2, y2 = coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img=image,
            text=text,
            org=(x1, y1-2),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale= img_height/2000,
            color=color,
            thickness=1
        )
        return None

    @staticmethod
    def crop_frame(image, coordinates, bb_box=True):
        if bb_box:
            x,y,w,h = coordinates
            return image[y:y+h, x:x+w]
        else:
            x1, y1, x2, y2 = coordinates
            return image[y1:y2, x1:x2]

    @staticmethod
    def save_crop(output_dir, image_name, cropped_img):
        name = os.path.join(output_dir, image_name)
        cv2.imwrite(name, cropped_img)




class FashionFrameDetection(FrameDetector):

    def __init__(self, image, face_coordinates, person_frame_coordinates, colours, output_dir=None, display=False):
        FrameDetector().__init__()
        self.image = image
        self.face_coordinates = face_coordinates
        self.person_frame_coordinates = person_frame_coordinates
        self.colors = colours
        self.display = display
        self.save = False
        self.output_dir = output_dir
        if output_dir:
            self.save = True
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    
    @staticmethod
    def resize_with_aspect_ratio(img, width=640, inter=cv2.INTER_AREA):
        h,w = img.shape[:2]
        r = width/float(w)
        dim = width, int(h*r)
        return cv2.resize(img, dim, interpolation=inter)
        
    def frame_detect(self):
        image = self.image
        image_c = self.image
        x,y,w,h = self.face_coordinates
        (b_x1, b_y1), (b_x2, b_y2) = self.person_frame_coordinates
        x1, y1, w1, h1 = self.neck_coordinates(x, y, w, h)
        x1_1, y1_1, w1_1, h1_1 = self.deep_neck_coordinates(x1, y1, w, h, h1)
        x2, y2, w2, h2 = self.upper_half_torso_coordinates(x, y, w, h, h1)
        x3, y3, w3, h3 = self.lower_half_torso_coordinates(x2, y2, w2, h2, h)
        x_l1, y_l1, x_l2, y_l2 = self.left_arm_coordinates(b_x1, y2, x3, y3, h3)
        x_lh1, y_lh1, x_lh2, y_lh2 = self.left_hand_coordinates(b_x1, x3, y3, h3, h)
        x_r1, y_r1, x_r2, y_r2 = self.right_arm_coordinates(b_x2, y2, x3, y3, h3, w3)
        x_rh1, y_rh1, x_rh2, y_rh2 = self.right_hand_coordinates(b_x2, x3, y3, h3, h, w3)
        x4, y4, x4_, y4_ = self.legs_coordinates(x3, y3, w3, h3, b_y2)
        x5, y5, x5_, y5_ = self.foot_coordinates(x4, x4_, y4_, h)
        x6, y6, x6_, y6_ = self.dress_fullness(b_x1, y2, h2, b_x2, b_y2)

        if self.save:
            text = "Person"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(b_x1, b_y1, b_x2, b_y2), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)
            text = "Neck"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x1, y1, w1, h1))
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)
            text = "Deep-Neck"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x1_1, y1_1, w1_1, h1_1))
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)
            text = "Upper-Half-Torso"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x2, y2, w2, h2))
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)
            text = "Lower-Half-Torso"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x3, y3, w3, h3))
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)

            # left limbs
            text = "Left Arm"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x_l1, y_l1, x_l2, y_l2), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{'-'.join(text.split()).lower()}.jpg", cropped_img=cropped_frame)
            text = "Left Hand"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x_lh1, y_lh1, x_lh2, y_lh2), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{'-'.join(text.split()).lower()}.jpg", cropped_img=cropped_frame)

            # right limbs
            text = "Right Arm"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x_r1, y_r1, x_r2, y_r2), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{'-'.join(text.split()).lower()}.jpg", cropped_img=cropped_frame)
            text = "Right Hand"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x_rh1, y_rh1, x_rh2, y_rh2), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{'-'.join(text.split()).lower()}.jpg", cropped_img=cropped_frame)

            text = "Legs"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x4, y4, x4_, y4_), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)
            text = "Feet"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x5, y5, x5_, y5_), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)
            text = "Fullness"
            cropped_frame = self.crop_frame(image=image_c, coordinates=(x6, y6, x6_, y6_), bb_box=False)
            self.save_crop(output_dir=self.output_dir, image_name=f"{text.lower()}.jpg", cropped_img=cropped_frame)

        if self.display:
            text = "Person"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (b_x1, b_y1, b_x2, b_y2), color_, text, bb_box=False)
            text = "Face"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, self.face_coordinates, color_, text)
            text = "Neck"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x1, y1, w1, h1), color_, text)
            text = "Deep-Neck"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x1_1, y1_1, w1_1, h1_1), color_, text)
            text = "Upper-Half-Torso"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x2, y2, w2, h2), color_, text)
            text = "Lower-Half-Torso"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x3, y3, w3, h3), color_, text)

            # left limbs
            text = "Left Arm"
            color_ = self.colors["arm"]
            self.draw_rectangle(image, (x_l1, y_l1, x_l2, y_l2), color_, text, bb_box=False)
            text = "Left Hand"
            color_ = self.colors["hand"]
            self.draw_rectangle(image, (x_lh1, y_lh1, x_lh2, y_lh2), color_, text, bb_box=False)

            # right limbs
            text = "Right Arm"
            color_ = self.colors["arm"]
            self.draw_rectangle(image, (x_r1, y_r1, x_r2, y_r2), color_, text, bb_box=False)
            text = "Right Hand"
            color_ = self.colors["hand"]
            self.draw_rectangle(image, (x_rh1, y_rh1, x_rh2, y_rh2), color_, text, bb_box=False)

            text = "Legs"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x4, y4, x4_, y4_), color_, text, bb_box=False)
            text = "Feet"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x5, y5, x5_, y5_), color_, text, bb_box=False)
            text = "Fullness"
            color_ = self.colors[text.lower()]
            self.draw_rectangle(image, (x6, y6, x6_, y6_), color_, text, bb_box=False)

        # saving final image
        self.save_crop(output_dir=self.output_dir, image_name="view.jpg", cropped_img=image)
        # image = self.resize_with_aspect_ratio(img=image)

        return image




