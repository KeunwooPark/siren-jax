import argparse
import numpy as np

from siren.model import get_model_cls_by_type
from util.log import Loader, Logger
from siren.data_loader import convert_to_normalized_index, unnormalize_img, xy_to_image_array, split_to_batches
from PIL import Image
from util.image import gradient_to_img

def parse_args():
    parser = argparse.ArgumentParser(description="Test SirenHighres")

    parser.add_argument('--run_name', type=str, help="the name of a train run")
    parser.add_argument('--size', type=int, default=512, help="size of image to generate. set to 0 if to use the original image size.")
    
    args = parser.parse_args()
    return args

def main(args):
    loader = Loader(args.run_name)
    logger = Logger(args.run_name, create_if_exists=False)
    option = loader.load_option()
    layers = [int(l) for l in option['layers'].split(',')]
    params = loader.load_params()
    
    Model = get_model_cls_by_type(option['type'])
    model = Model(layers, option['nc'], option['omega'])
    model.update_net_params(params)
    
    if args.size == 0:
        orig_img_fn = loader.get_image_filename("original")
        img = Image.open(orig_img_fn)
        width = img.width
        height = img.height
    else:
        width = args.size
        height = args.size


    estimate_and_save_image(model, width, height, logger)
    if option['nc'] == 1:
        estimate_and_save_gradient(model, width, height, logger)

    
    if option['type'] == 'normal':
        # PIL resize as reference
        input_pil_img = loader.load_pil_image("input")
        resized_pil = input_pil_img.resize((width, height))
        pil_output_name = "pil_{}x{}".format(width, height)
        logger.save_image(pil_output_name, resized_pil)

def estimate_and_save_image(model, width, height, logger):
    x = convert_to_normalized_index(width, height)
    
    batched_x, _ = split_to_batches(x, size=2048)
    batched_y = []
    for bx in batched_x:
        y = model.forward(bx)
        
        batched_y.append(y)

    y = np.vstack(batched_y)

    # save normal image
    img = xy_to_image_array(x, y, width, height)
    img = unnormalize_img(img)

    output_name = "net_{}x{}".format(width, height)
    logger.save_image(output_name, img)

def estimate_and_save_gradient(model, width, height, logger):
    x = convert_to_normalized_index(width, height)
    
    batched_x, _ = split_to_batches(x, size=2048)
    batched_y = []
    for bx in batched_x:
        y = model.gradient(bx)
        
        batched_y.append(y)

    y = np.vstack(batched_y)
    y = y.squeeze()
    y = xy_to_image_array(x, y, width, height)
    grad_img = gradient_to_img(y)
    output_name = "grad_{}x{}".format(width, height)
    logger.save_image(output_name, grad_img)

if __name__ == "__main__":
    args = parse_args()
    main(args)
