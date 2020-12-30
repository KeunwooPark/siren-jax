from scipy import ndimage
import numpy as np

def gradient(img):
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=2)
    
    return np.concatenate([grad_x, grad_y], axis = -1)

def gradient_to_img(gradient):
    pass
