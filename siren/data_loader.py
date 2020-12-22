from PIL import Image
import numpy as np

class ImageLoader:
    def __init__(self, img_path, size=0, do_batch=False, batch_size=256):
        img = Image.open(img_path)
        
        if size > 0:
            img = img.resize((size, size))

        self.normalized_img = normalize_img(np.array(img))
         
        self.do_batch = do_batch
        self.batch_size = batch_size

        self.x, self.y = image_to_xy(self.normalize_img)

    def get(self, i):
        pass

def normalize_img(img_array):
    return (img_array - 0.5) / 0.5

def unnormalize_img(img_array):
    return img_array * 0.5 + 0.5

def image_to_xy(img_array):
    width, height, channel = img_array.shape
    x = []
    y = []

    for i in range(width):
        for j in range(height):
            x.append([i, j])
            y.append(img_array[i, j])
    return np.array(x), np.array(y)

def split_to_batches(x, y, size = 0):
    if size == 0:
        num_batches = 1
        return [x], [y], num_batches

    num_sample = x.shape[0]
    num_batches = num_sample // size
    if not num_sample % size == 0:
        num_batches += 1

    batched_x = np.split(x, num_batches)
    batched_y = np.split(y, num_batches)

    return batched_x, batched_y, num_batches
