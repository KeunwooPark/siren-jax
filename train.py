import argparse
from siren.data_loader import ImageLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train SirenHighres')
    
    parser.add_argument('--file', type=str, help="location of the file")
    parser.add_argument('--size', type=int, default=256, help="resize the image to this (squre) shape. 0 if not goint go resize")
    parser.add_argument('--do_batch', action='store_true', default=False, help="separate input to batches")
    parser.add_argument('--batch_size', type=int, default=256, help="the size of batches. only valid when --do_batch")
    parser.add_argument('--epoch', type=int, default=1000, help="number of epochs")
    
    args = parser.parse_args()
    return args

def main(args):
    image_loader = ImageLoader(args.file, args.downsample, args.do_batch, args.batch_size)
    print(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
