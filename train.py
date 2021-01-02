import argparse
from siren.data_loader import get_data_loader_cls_by_type
from siren.optimizer import JaxOptimizer
from siren.model import get_model_cls_by_type
from util.log import Logger
from util.timer import Timer

def parse_args():
    parser = argparse.ArgumentParser(description='Train SirenHighres')

    parser.add_argument('--file', type=str, help="location of the file", required=True)
    parser.add_argument('--nc', type=int, default=3, help="number of channels of input image. if the source is color (3) and --nc is 1, then the source is converted to gray scale")
    parser.add_argument('--type', type=str, default="normal", choices=["normal", "gradient", "laplacian"], help="training image type")
    parser.add_argument('--size', type=int, default=256, help="resize the image to this (squre) shape. 0 if not goint go resize")
    parser.add_argument('--batch_size', type=int, default=16384, help="the size of batches. 0 for single batch")
    parser.add_argument('--epoch', type=int, default=10000, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--print_iter', type=int, default=200, help="when to print intermediate info")
    parser.add_argument('--layers', type=str, default='256,256,256', help="layers of multi layer perceptron")
    parser.add_argument('--omega', type=float, default=30, help="omega value of Siren")

    args = parser.parse_args()
    return args

def main(args):
    layers = [int(l) for l in args.layers.split(',')]
    
    Model = get_model_cls_by_type(args.type)
    DataLoader = get_data_loader_cls_by_type(args.type)

    data_loader = DataLoader(args.file, args.nc, args.size, args.batch_size)
    model = Model(layers, args.nc, args.omega)
    optimizer = JaxOptimizer('adam', model, args.lr)

    name = args.file.split('.')[0]
    logger = Logger(name)
    logger.save_option(vars(args))
    
    gt_img = data_loader.get_ground_truth_image()
    logger.save_image("original", data_loader.original_pil_img)
    logger.save_image("gt", gt_img)

    iter_timer = Timer()
    iter_timer.start()
    def interm_callback(i, data, params):
        log = {}
        loss = model.loss_func(params, data)
        log['loss'] = float(loss)
        log['iter'] = i
        log['duration_per_iter'] = iter_timer.get_dt() / args.print_iter

        logger.save_log(log)
        print(log)

    print("Training Start")
    print(vars(args))
    
    total_timer = Timer()
    total_timer.start()
    last_data = None
    for _ in range(args.epoch):
        data_loader = DataLoader(args.file, args.nc, args.size, args.batch_size)
        for data in data_loader:
            optimizer.step(data)
            last_data = data
            if optimizer.iter_cnt % args.print_iter == 0:
                interm_callback(optimizer.iter_cnt, data, optimizer.get_optimized_params())


    if not optimizer.iter_cnt % args.print_iter == 0:
        interm_callback(optimizer.iter_cnt, data, optimizer.get_optimized_params())

    train_duration = total_timer.get_dt()
    print("Training Duration: {} sec".format(train_duration))
    logger.save_net_params(optimizer.get_optimized_params())
    logger.save_losses_plot()

if __name__ == "__main__":
    args = parse_args()
    main(args)
