from PIL import Image
import numpy as np
from jax import numpy as jnp

class ImageLoader:
    def __init__(self, img_path, size=0, do_batch=False, batch_size=256):
        img = Image.open(img_path)

        if size > 0:
            img = img.resize((size, size))

        self.normalized_img = normalize_img(np.array(img))

        self.do_batch = do_batch
        self.batch_size = batch_size

        self.x, self.y = image_to_xy(self.normalized_img)
        self.create_batches()
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = self.get(self.cursor)
        except IndexError:
            raise StopIteration

        self.cursor += 1
        return data

    def __len__(self):
        return self.num_batches

    def create_batches(self):
        self.batched_x, self.batched_y, self.num_batches = split_to_batches(self.x, self.y, size = self.batch_size)

    def get(self, i):
        x = jnp.array(self.batched_x[i])
        y = jnp.array(self.batched_y[i])
        data = {'input': x, 'output': y}
        return data

def normalize_img(img_array):
    return (img_array - 0.5) / 0.5

def unnormalize_img(img_array):
    return img_array * 0.5 + 0.5

def convert_to_normalized_index(width, height):
    normalized_index = []
    for i in np.linspace(-1, 1, width):
        for j in np.linspace(-1, 1, height):
            normalized_index.append([i, j])

    return np.array(normalized_index)

def image_to_xy(img_array):
    width, height, channel = img_array.shape
    y = []

    x = convert_to_normalized_index(width, height)

    for i in range(width):
        for j in range(height):
            y.append(img_array[i, j])
    return x, np.array(y)

def split_to_batches(x, y, size = 0, shuffle=True):
    if shuffle:
        x = shuffle_array(x)
        y = shuffle_array(y)

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

def shuffle_array(array):
    size = array.shape[0]
    indicies = np.arange(size, dtype=np.int)
    np.random.shuffle(indicies)
    return array[indicies]
