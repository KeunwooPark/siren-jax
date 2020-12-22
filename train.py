import argparse
from siren.data_loader import ImageLoader
from siren.optimizer import minimize_with_jax_optim
from siren.model import ImageModel
from util.log import Logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train SirenHighres')

    parser.add_argument('--file', type=str, help="location of the file", required=True)
    parser.add_argument('--size', type=int, default=256, help="resize the image to this (squre) shape. 0 if not goint go resize")
    parser.add_argument('--do_batch', action='store_true', default=False, help="separate input to batches")
    parser.add_argument('--batch_size', type=int, default=256, help="the size of batches. only valid when --do_batch")
    parser.add_argument('--epoch', type=int, default=1, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--print_iter', type=int, default=1000, help="when to print intermediate info")
    parser.add_argument('--layers', type=str, default='128,128,128', help="layers of multi layer perceptron")

    args = parser.parse_args()
    return args

def main(args):
    layers = [int(l) for l in args.layers.split(',')]
    model = ImageModel(layers)
    image_loader = ImageLoader(args.file, args.size, args.do_batch, args.batch_size)
    name = args.file.split('.')[0]
    logger = Logger(name)
    logger.save_option(vars(args))

    def callback(data, training_state):
        log = {}
        loss_func = model.get_loss_func(data)
        log['loss'] = loss_func(training_state.params)
        log['iter'] = training_state.iter
        log['duration_per_iter'] = training_state.duration_per_iter
        logger.save_log(log)

    minimize_with_jax_optim('adam', model, image_loader, args.lr, args.epoch, args.print_iter, callback)
     

if __name__ == "__main__":
    args = parse_args()
    main(args)
