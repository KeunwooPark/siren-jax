import argparse

from siren.model import ImageModel
from util.log import Loader

def parse_args():
    parser = argparse.ArgumentParser(description="Test SirenHighres")

    parser.add_argument('--run_name', type=str, help="the name of a train run")
    parser.add_argument('--comp_orig', action="store_true", help="generate original image size and calculate loss with the original image")
    parser.add_argument('--size', type=int, default=512, help="size of image to generate. ignored if --comp_orig")
    
    args = parser.parse_args()
    return args

def main(args):
    loader = Loader(args.run_name)
    option = loader.load_option()
    layers = [int(l) for l in option['layers'].split(',')]
    params = loader.load_params()

    model = ImageModel(layers)
    model.update_net_params(params)
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
