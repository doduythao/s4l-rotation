import argparse
import os

import torchvision

from rot_evaluator import Evaluator
from rot_model import Model, Model2
from wide_model import Wide_ResNet

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logdir', default='./logs', help='directory to write logs')
parser.add_argument('-r', '--restore_checkpoint', default=None, help='path to restore checkpoint')
parser.add_argument('-impr', '--is_improve', default=False, type=bool, help='improved version or not')
parser.add_argument('-m', '--wideresnet', default=False, type=bool, help='wide resnet or not')


def _eval(train_opts):
    checkpoint_file = train_opts['restore_checkpoint']

    if not train_opts['is_improve']:
        model = Model()
    else:
        model = Model2()
    
    if train_opts['wideresnet']:
        model = Wide_ResNet(28, 2, 0.2, 10)
    
    model.cuda()

    # load checkpoints
    if checkpoint_file is not None:
        assert os.path.isfile(checkpoint_file), '%s not found' % checkpoint_file
        model.restore(checkpoint_file)
        print('Model restored from file: %s' % checkpoint_file)

    # prepare datasets
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    evaluator = Evaluator(test_set)
    return evaluator.evaluate(model)


def main(args):
    path_to_log_dir = args.logdir
    if not os.path.exists(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    options = {
        'logdir': args.logdir,
        'restore_checkpoint': args.restore_checkpoint,
        'is_improve': args.is_improve,
        'wideresnet': args.wideresnet
    }

    print('Start evaluate:', _eval(options))
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
