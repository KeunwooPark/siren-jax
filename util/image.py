from scipy import ndimage
import numpy as np
import matplotlib.colors as colors

def gradient(img):
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    
    return np.concatenate([grad_x, grad_y], axis = -1)

def rescale_img(x, max_val = 1.0, min_val = 0.0):
    x_min = np.min(x)
    x_max = np.max(x)

    rescaled = (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val

    return rescaled

def gradient_to_img(gradient):
    # code from original Siren implementation
    # https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/dataio.py#L55

    n_rows = gradient.shape[0]
    n_cols = gradient.shape[1]

    gx = gradient[:, :, 1]
    gy = gradient[:, :, 0]

    ga = np.arctan2(gy, gx)
    gm = np.hypot(gy, gx)

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
