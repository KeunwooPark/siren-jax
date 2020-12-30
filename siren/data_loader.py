from PIL import Image, ImageOps
import numpy as np
from jax import numpy as jnp
from util.image import gradient, gradient_to_img

def get_data_loader_cls_by_type(type):
    if type == 'color':
        return ColorImageLoader
    elif type == 'gradient':
        return GradientImageLoader

    raise ValueError("Wrong data loader type: {}".format(type))

class ColorImageLoader:
    def __init__(self, img_path, size=0, batch_size=0):
        img = Image.open(img_path)
        self.original_pil_img = img

        if size > 0:
            img = img.resize((size, size))

        self.input_img = normalize_img(np.array(img))

        self.do_batch = batch_size != 0
        self.batch_size = batch_size

        self.x, self.y = image_array_to_xy(self.input_img)
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
        if not self.do_batch:
            self.batched_x, self.num_batches = split_to_batches(self.x, size=0)
            self.batched_y, self.num_batches = split_to_batches(self.y, size=0)
        else:
            shuffled_x, shuffled_y = shuffle_arrays_in_same_order([self.x, self.y])
            self.batched_x, self.num_batches = split_to_batches(self.x, size = self.batch_size)
            self.batched_y, self.num_batches = split_to_batches(self.y, size = self.batch_size)


    def get(self, i):
        x = jnp.array(self.batched_x[i])
        y = jnp.array(self.batched_y[i])
        data = {'input': x, 'output': y}
        return data

    def get_input_image(self):
        img = unnormalize_img(self.input_img)
        return Image.fromarray(np.uint8(img))

class GradientImageLoader(ColorImageLoader):
    def __init__(self, img_path, size=0, batch_size=0):
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        self.original_pil_img = img

        if size > 0:
            img = img.resize((size, size))

        img = np.expand_dims(np.array(img), axis=-1)
        img = normalize_img(img)
        self.input_img = gradient(img)

        self.do_batch = batch_size != 0
        self.batch_size = batch_size

        self.x, self.y = image_array_to_xy(self.input_img)
        self.create_batches()
        self.cursor = 0

    def get_input_image(self):

        img = gradient_to_img(self.input_img) 
        img = img * 255
        img = img.squeeze()
        return Image.fromarray(np.uint8(img))
        
def normalize_img(img_array):
    img_array = img_array / 255
    return (img_array - 0.5) / 0.5

def unnormalize_img(img_array):
    return (img_array * 0.5 + 0.5) * 255

def convert_to_normalized_index(width, height):
    normalized_index = []
    for i in np.linspace(-1, 1, width):
        for j in np.linspace(-1, 1, height):
            normalized_index.append([i, j])

    return np.array(normalized_index)

def image_array_to_xy(img_array):
    width, height, channel = img_array.shape
    y = []

    x = convert_to_normalized_index(width, height)

    for i in range(width):
        for j in range(height):
            y.append(img_array[i, j])
    return x, np.array(y)

def xy_to_image_array(x, y, width, height):
    w_idx = ((x[:, 0] + 1) / 2) * (width-1)
    h_idx = ((x[:, 1] + 1) / 2) * (height-1)

    w_idx = np.around(w_idx).astype(np.int)
    h_idx = np.around(h_idx).astype(np.int)

    #ww, hh = np.meshgrid(w_idx, h_idx)
    
    num_channel=y.shape[-1]
    img_array = np.zeros((width, height, num_channel))
    
    img_array[w_idx, h_idx] = y

    return img_array
        

def split_to_batches(array, size = 0):
    if size == 0:
        num_batches = 1
        return [array], num_batches

    num_sample = array.shape[0]
    num_batches = int(np.ceil(num_sample / size))
    batched = np.array_split(array, num_batches)

    return batched, num_batches

def shuffle_arrays_in_same_order(arrays):
    size = arrays[0].shape[0]
    indicies = np.arange(size, dtype=np.int)

    np.random.shuffle(indicies)

    shuffled_arrays = []
    for array in arrays:
        shuffled_arrays.append(array[indicies])

    return shuffled_arrays
