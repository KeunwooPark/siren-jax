from scipy import ndimage
import numpy as np
import matplotlib.colors as colors
import cv2
import cmapy

def gradient(img):
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    
    return np.concatenate([grad_y, grad_x], axis = -1)

def laplace(img):
    return ndimage.laplace(img)

def rescale_img(x, max_val = 1.0, min_val = 0.0):
    x_min = np.min(x)
    x_max = np.max(x)

    rescaled = (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val

    return rescaled

def clip_img_by_perc(x, perc):
    x_min = np.percentile(x, perc)
    x_max = np.percentile(x, 100 - perc)
    x = np.clip(x, x_min, x_max)

    return x

def gradient_to_img(gradient):
    # code from original Siren implementation
    # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/dataio.py#L55

    n_rows = gradient.shape[0]
    n_cols = gradient.shape[1]

    gx = gradient[:, :, 1]
    gy = gradient[:, :, 0]

    ga = np.arctan2(gx, gy)
    gm = np.hypot(gx, gy)

    hsv = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    hsv[:, :, 0] = (ga + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = 1.

    gm_min = np.percentile(gm, 5)
    gm_max = np.percentile(gm, 95)
    
    gm = (gm - gm_min) / (gm_max - gm_min)
    gm = np.clip(gm, 0, 1)

    hsv[:, :, 2] = gm
    rgb = colors.hsv_to_rgb(hsv)
    rgb *= 255
    return rgb

def laplace_to_img(laplace):
    laplace = clip_img_by_perc(laplace, 2)
    rescaled = rescale_img(laplace, max_val=255, min_val=0)
    
    colormap_img = cv2.applyColorMap(np.uint8(rescaled).squeeze(), cmapy.cmap('RdBu'))
    img = cv2.cvtColor(colormap_img, cv2.COLOR_BGR2RGB)
    return img
