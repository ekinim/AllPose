import cv2
import numpy as np


def pad_image(image):
    H, W, D = image.shape
    diff = W - H
    dtype = 'uint8'
    
    if diff < 0:
        left_pad_size = abs(diff) // 2
        right_pad_size = H - W - left_pad_size
        
        left_padding = np.zeros((H, left_pad_size, D), dtype=dtype)
        right_padding = np.zeros((H, right_pad_size, D), dtype=dtype)
        image = np.concatenate([left_padding, image, right_padding], axis=1)

    elif diff > 0:
        upper_pad_size = abs(diff) // 2
        lower_pad_size = W - H - upper_pad_size
        
        upper_padding = np.zeros((upper_pad_size, W, D), dtype=dtype)
        lower_padding = np.zeros((lower_pad_size, W, D), dtype=dtype)
        image = np.concatenate([upper_padding, image, lower_padding], axis=0)

    return image


def print_joints(image, coords):

    W = image.shape[1]
    H = image.shape[0]
    
    for i, person in enumerate(coords):
        for j, (x, y, z) in enumerate(person):
            x = int(x*H)
            y = int(y*W)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 0.4
            color = (0,0,0) 
            thickness = 1
            image = cv2.putText(image, str(j), (y,x), font, fontScale, color, thickness, cv2.LINE_AA)

    return image