from PIL import Image
import numpy as np
from jax import numpy as jnp
from util.image import gradient, gradient_to_img, laplace, laplacian_to_img
from abc import ABC, abstractmethod

def get_data_loader_cls_by_type(type):
    if type == 'normal':
        return NormalImageLoader
    elif type == 'gradient':
        return GradientImageLoader
    elif type == 'laplacian':
        return LaplacianImageLoader

    raise ValueError("Wrong data loader type: {}".format(type))

class BaseImageLoader(ABC):
    def __init__(self, img_path, num_channels, size=0, batch_size=0):
        img = Image.open(img_path)

        if size > 0:
            img = img.resize((size, size))

        if num_channels == 3:
            img = img.convert("RGB")
            img_array = np.array(img)
        elif num_channels == 1:
            img = img.convert("L")
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis = -1)
        else:
            raise ValueError("Wrong number of channels")

        self.original_pil_img = img
        #self.input_img = normalize_img(img_array)
        self.gt_img = self.create_ground_truth_img(img_array)
     
        self.do_batch = batch_size != 0
        self.batch_size = batch_size

        self.x, self.y = image_array_to_xy(self.gt_img)
        self.create_batches()
        self.cursor = 0
        
    @abstractmethod
    def create_ground_truth_img(self):
        pass
            
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

    @abstractmethod
    def get_ground_truth_image(self):
        pass
        
class NormalImageLoader(BaseImageLoader):
    def create_ground_truth_img(self, img_array):
        return normalize_img(img_array)

    def get_ground_truth_image(self):
        img = unnormalize_img(self.gt_img)
        img = img.squeeze()
        return Image.fromarray(np.uint8(img))

class GradientImageLoader(BaseImageLoader):
    def create_ground_truth_img(self, img_array):
        img = normalize_img(img_array)
        return gradient(img * 1e1)

    def get_ground_truth_image(self):
        img = gradient_to_img(self.gt_img)
        img = img.squeeze()
        return Image.fromarray(np.uint8(img))

class LaplacianImageLoader(BaseImageLoader):
    def create_ground_truth_img(self, img_array):
        img = normalize_img(img_array)
        return laplace(img * 1e4)

    def get_ground_truth_image(self):
        img = laplacian_to_img(self.gt_img)
        return Image.fromarray(np.uint8(img))
        
def normalize_img(img_array):
    img_array = img_array / 255.0
    return (img_array - 0.5) / 0.5

def unnormalize_img(img_array):
    return (img_array * 0.5 + 0.5) * 255

def convert_to_normalized_index(width, height):
    normalized_index = []
    i = np.linspace(-1, 1, width)
    j = np.linspace(-1, 1, height)
    ii, jj = np.meshgrid(i, j, indexing='ij')

    normalized_index = np.stack([ii, jj], axis = -1)
    return np.reshape(normalized_index, (-1, 2))

def image_array_to_xy(img_array):
    width, height, channel = img_array.shape

    x = convert_to_normalized_index(width, height)
    num_channel = img_array.shape[-1]
    y = np.reshape(img_array, (-1, num_channel))
    return x, np.array(y)

def xy_to_image_array(x, y, width, height):
    w_idx = ((x[:, 0] + 1) / 2) * (width-1)
    h_idx = ((x[:, 1] + 1) / 2) * (height-1)

    w_idx = np.around(w_idx).astype(np.int)
    h_idx = np.around(h_idx).astype(np.int)
                            
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
