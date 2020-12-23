import argparse
import numpy as np

from siren.model import ImageModel
from util.log import Loader, Logger
from siren.data_loader import convert_to_normalized_index, unnormalize_img, xy_to_image_array, split_to_batches
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Test SirenHighres")

    parser.add_argument('--run_name', type=str, help="the name of a train run")
    parser.add_argument('--comp_orig', action="store_true", help="generate original image size and calculate loss with the original image")
    parser.add_argument('--size', type=int, default=512, help="size of image to generate. ignored if --comp_orig")
    
    args = parser.parse_args()
    return args

def main(args):
    loader = Loader(args.run_name)
    logger = Logger(args.run_name, create_if_exists=False)
    option = loader.load_option()
    layers = [int(l) for l in option['layers'].split(',')]
    params = loader.load_params()

    model = ImageModel(layers, option['omega'])
    model.update_net_params(params)
    
    if args.comp_orig:
        orig_img_fn = loader.get_image_filename("original")
        img = Image.open(orig_img_fn)
        width = img.width
        height = img.height
    else:
        width = args.size
        height = args.size

    x = convert_to_normalized_index(width, height)
    
    # divide input to batches, because usually input for the test is really large
    batched_x, _ = split_to_batches(x, size=2048)
    batched_y = []
    for bx in batched_x:
        y = model.forward(bx)
        batched_y.append(y)

    y = np.vstack(batched_y)

    img_array = xy_to_image_array(x, y, width, height)
    img_array = unnormalize_img(img_array)
    img = Image.fromarray(np.uint8(img_array))
    
    output_name = "net_{}x{}".format(width, height)
    logger.save_image(output_name, img)

    # PIL resize as reference
    input_pil_img = loader.load_pil_image("input")
    resized_pil = input_pil_img.resize((width, height))
    pil_output_name = "pil_{}x{}".format(width, height)
    logger.save_image(pil_output_name, resized_pil)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
