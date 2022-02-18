from curses import window
import cv2
import numpy as np
from typing import TypedDict


def imshow(window_name, data, wait=False):
    '''
    Utility function to show images to the user
    '''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, data)

    if wait:
        cv2.waitKey(0)