import argparse
from siren.data_loader import ImageLoader
from siren.optimizer import JaxOptimizer
from siren.model import ImageModel
from util.log import Logger
from util.timer import Timer

def parse_args():
    parser = argparse.ArgumentParser(description='Train SirenHighres')

    parser.add_argument('--file', type=str, help="location of the file", required=True)
    parser.add_argument('--size', type=int, default=256, help="resize the image to this (squre) shape. 0 if not goint go resize")
    parser.add_argument('--do_batch', action='store_true', default=False, help="separate input to batches")
    parser.add_argument('--batch_size', type=int, default=256, help="the size of batches. only valid when --do_batch")
    parser.add_argument('--epoch', type=int, default=1000, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--print_iter', type=int, default=100, help="when to print intermediate info")
    parser.add_argument('--layers', type=str, default='256,256,256,256,256', help="layers of multi layer perceptron")

    args = parser.parse_args()
    return args

def main(args):
    layers = [int(l) for l in args.layers.split(',')]
    model = ImageModel(layers)
    image_loader = ImageLoader(args.file, args.size, args.do_batch, args.batch_size)
    optimizer = JaxOptimizer('adam', model, args.lr)

    name = args.file.split('.')[0]
    logger = Logger(name)
    logger.save_option(vars(args))
    
    input_img = image_loader.get_resized_image()
    logger.save_image("input", input_img)

    timer = Timer()
    timer.start()
    def interm_callback(i, data, params):
        log = {}
        loss = model.loss_func(params, data)
        log['loss'] = float(loss)
        log['iter'] = i
        log['duration_per_iter'] = timer.get_dt() / args.print_iter

        logger.save_log(log)
        print(log)

    print("Training Start")
    print(vars(args))

    last_data = None
    for _ in range(args.epoch):
        image_loader = ImageLoader(args.file, args.size, args.do_batch, args.batch_size)
        for data in image_loader:
            optimizer.step(data)
            last_data = data
            if optimizer.iter_cnt % args.print_iter == 0:
                interm_callback(optimizer.iter_cnt, data, optimizer.get_optimized_params())


    if not optimizer.iter_cnt % args.print_iter == 0:
        interm_callback(optimizer.iter_cnt, data, optimizer.get_optimized_params())

    logger.save_net_params(optimizer.get_optimized_params())
    logger.save_losses_plot()

if __name__ == "__main__":
    args = parse_args()
    main(args)
