import argparse
from siren.data_loader import ImageLoader
from siren.optimizer import minimize_with_jax_optim
from siren.model import ImageModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train SirenHighres')

    parser.add_argument('--file', type=str, help="location of the file", required=True)
    parser.add_argument('--size', type=int, default=256, help="resize the image to this (squre) shape. 0 if not goint go resize")
    parser.add_argument('--do_batch', action='store_true', default=False, help="separate input to batches")
    parser.add_argument('--batch_size', type=int, default=256, help="the size of batches. only valid when --do_batch")
    parser.add_argument('--epoch', type=int, default=1, help="number of epochs")
    parser.add_argument('--layers', type=str, default='128,128,128', help="layers of multi layer perceptron")

    args = parser.parse_args()
    return args

def main(args):
    layers = [int(l) for l in args.layers.split(',')]
    model = ImageModel(layers)
    image_loader = ImageLoader(args.file, args.size, args.do_batch, args.batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)
