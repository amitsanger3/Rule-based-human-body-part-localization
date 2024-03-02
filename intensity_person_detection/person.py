import cv2
import numpy as np
from rembg import remove


class IntensityPersonDetect(object):

    def __init__(self, image):
        self.image = image
        self.h, self.w, _= self.image.shape

    def remove_background(self):
        return remove(self.image)

    @staticmethod
    def find_all_indices(array):
        """
        Finds all indices of the first non-zero value and the first zero value after it.

        Args:
            array: A NumPy array of any data type.

        Returns:
            A list of tuples, where each tuple contains two elements:
                - The index of the first non-zero value in a segment.
                - The index of the first zero value after the first non-zero value,
                    or -1 if there are no non-zero values in that segment.
        """
        non_zero_starts = []
        non_zero_ends = []

        # Iterate through the array, keeping track of the current non-zero segment
        current_start = None
        for i, value in enumerate(array):
            if value != 0:
                if current_start is None:
                    current_start = i
            elif current_start is not None:
                non_zero_starts.append(current_start)
                non_zero_ends.append(i)
                current_start = None

        # Handle the last segment if it's non-zero
        if current_start is not None:
            non_zero_starts.append(current_start)
            non_zero_ends.append(-1)  # No ending zero for the last segment

        return list(zip(non_zero_starts, non_zero_ends))

    @staticmethod
    def modify_array(array, threshold=10):
        """
        Modifies the array by setting elements less than the threshold to 0.

        Args:
            array: A NumPy array of any data type.
            threshold: The value below which elements will be set to 0 (default 10).

        Returns:
            A new NumPy array with the modified elements.
        """
        return np.where(array < threshold, 0, array)

    @staticmethod
    def get_cordinates(all_indices, intensity_map, x_=True):
        cords = []
        for start, end in all_indices:
            if start == -1:
                print("No non-zero values found in a segment.")
            else:
                # print(f"Segment starting at {start}:")
                if end != -1:
                    # print(f"  - Index of the first non-zero value: {start}")
                    # print(f"  - Index of the first zero value after the first non-zero: {end}")
                    cords += [start, end]
                else:
                    # print(f"  - Index of the first non-zero value: {start}")
                    # print("  - No zero found after the first non-zero value.")
                    cords += [start, intensity_map.shape[0] - 3]
        return cords

    @staticmethod
    def binary_gray(img, reduction=10):
        return np.where(img - np.array([reduction]) < 0, 0, 1)

    def detect(self, threshold=10):
        no_bg_img = self.remove_background()
        no_bg_gray = cv2.cvtColor(no_bg_img, cv2.COLOR_BGR2GRAY)
        no_bg_gray = self.binary_gray(no_bg_gray)

        # on X axis
        intensity_map = no_bg_gray.sum(axis=0)
        all_indices = self.find_all_indices(self.modify_array(intensity_map, threshold=threshold))
        x_cords = self.get_cordinates(all_indices, intensity_map)

        # on y axis
        intensity_map = no_bg_gray.sum(axis=1)
        all_indices = self.find_all_indices(self.modify_array(intensity_map, threshold=threshold))
        y_cords = self.get_cordinates(all_indices, intensity_map)

        if len(x_cords) == 0 or len(y_cords) == 0:
            print('No model detected')
            return None

        return list(zip(x_cords, y_cords))

    @staticmethod
    def draw_bb(img, bb_list, first=True):
        if first:
            cv2.rectangle(img, bb_list[0], bb_list[1], (0, 255, 0), 3)
        else:
            cv2.rectangle(img, bb_list[0], bb_list[-1], (0, 255, 0), 3)
        return img
