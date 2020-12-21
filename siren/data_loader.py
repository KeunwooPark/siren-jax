from PIL import Image
import numpy as np

class ImageLoader:
    def __init__(self, img_path, do_batch=False, batch_size=256):
        img = Image.open(image_path)
        self.img = normalize_img(np.array(img))
        
        self.do_batch = do_batch
        self.batch_size = batch_size

    def get(self, idx):
        pass


def normalize_img(img_array):
    return (img_array - 0.5) / 0.5

def unnormalize_img(img_array):
    return img_array * 0.5 + 0.5

def image_to_xy(img_array):
    pass
