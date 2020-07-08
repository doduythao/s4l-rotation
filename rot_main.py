import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader

from rot_dataset import LabDataset, RotDataset
from rot_model import Model
from wide_model import Wide_ResNet
from rot_evaluator import Evaluator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logdir', default='./logs', help='directory to write logs')
parser.add_argument('-r', '--restore_checkpoint', default=None, help='path to restore checkpoint')
parser.add_argument('-bs', '--batch_size', default=32, type=int, help='Batch size of label set')
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Default 1e-2')
parser.add_argument('-p', '--patience', default=50, type=int, help='Default 50, set -1 to train infinitely')
parser.add_argument('-ds', '--decay_steps', default=5000, type=int, help='Default 5000')
parser.add_argument('-dr', '--decay_rate', default=0.95, type=float, help='Default 0.95')
parser.add_argument('-ls', '--label_size', default=1000, type=int)
parser.add_argument('-lur', '--label_unlabel_ratio', default=2, type=int)
parser.add_argument('-ba', '--best_acc', default=0.0, type=float, help='Default 0.0')
parser.add_argument('-w', '--w', default=0.5, type=float)
parser.add_argument('-vt', '--val_test', default=False, type=str2bool)
parser.add_argument('-lt', '--label_original', default=False, type=str2bool)
parser.add_argument('-m', '--wideresnet', default=False, type=str2bool)


# def imshow(img):
#     img = img / 2 + 0.5  # un normalize
#     npimg = img.numpy()
#     plt.figure()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))


def _loss(preds, ys, w, is_original):
    L_l, L_u = 0, 0
    for i in range(0, 4):
        L_l += F.cross_entropy(preds[i], ys[i])
    L_l = L_l / 4
    
    for i in range(4, 8):
        L_u += F.cross_entropy(preds[i], ys[i])
    L_u = L_u / 4
    
    return L_l + w * L_u


def _train(train_opts):
    batch_size = train_opts['batch_size']
    init_lr = train_opts['learning_rate']
    init_patience = train_opts['patience']
    checkpoint_file = train_opts['restore_checkpoint']
    log_dir = train_opts['logdir']
    nsteps_show_loss = 100
    nsteps_check = 1000

    step = 0
    patience = init_patience
    best_accuracy = train_opts['best_acc']
    if train_opts['wideresnet']:
        print('Use Wide ResNet')
        model = Wide_ResNet(28, 2, 0.2, 10)
    else:
        model = Model()
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=train_opts['decay_steps'], gamma=train_opts['decay_rate'])

    # load checkpoints
    if checkpoint_file is not None:
        assert os.path.isfile(checkpoint_file), '%s not found' % checkpoint_file
        step = model.restore(checkpoint_file)
        scheduler.last_epoch = step
        print('Model restored from file: %s' % checkpoint_file)

    # load losses report
    losses_file = os.path.join(log_dir, 'losses.npy')
    if os.path.isfile(losses_file):
        losses = np.load(losses_file)
    else:
        losses = np.empty([0], dtype=np.float32)

    # prepare datasets
    lab_size = train_opts['label_size']
    lu_ratio = train_opts['label_unlabel_ratio']
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    lab_set, unl_set, val = random_split(train_set, (
        lab_size, lab_size * lu_ratio, len(train_set) - lab_size * (lu_ratio + 1)))

    lab_set_ = LabDataset(lab_set)
#     lab_set0 = RotDataset(lab_set, 0)
    lab_set1 = RotDataset(lab_set, 1)
    lab_set2 = RotDataset(lab_set, 2)
    lab_set3 = RotDataset(lab_set, 3)

    unl_set0 = RotDataset(unl_set, 0)
    unl_set1 = RotDataset(unl_set, 1)
    unl_set2 = RotDataset(unl_set, 2)
    unl_set3 = RotDataset(unl_set, 3)

    if train_opts['val_test']:
        evaluator = Evaluator(val)
    else:
        evaluator = Evaluator(test_set)

    # Set batch size proportional to label & unlabelled ratio.
    lab_dl_ = DataLoader(lab_set_, batch_size=batch_size, shuffle=True)
#     lab_dl0 = DataLoader(lab_set0, batch_size=batch_size, shuffle=False)
    lab_dl1 = DataLoader(lab_set1, batch_size=batch_size, shuffle=True)
    lab_dl2 = DataLoader(lab_set2, batch_size=batch_size, shuffle=True)
    lab_dl3 = DataLoader(lab_set3, batch_size=batch_size, shuffle=True)

    unl_dl0 = DataLoader(unl_set0, batch_size=batch_size * lu_ratio, shuffle=True)
    unl_dl1 = DataLoader(unl_set1, batch_size=batch_size * lu_ratio, shuffle=True)
    unl_dl2 = DataLoader(unl_set2, batch_size=batch_size * lu_ratio, shuffle=True)
    unl_dl3 = DataLoader(unl_set3, batch_size=batch_size * lu_ratio, shuffle=True)

    duration = 0.0
    while True:
        for lab_bat, lab1_bat, lab2_bat, lab3_bat, unl0_bat, unl1_bat, unl2_bat, unl3_bat in zip(lab_dl_, lab_dl1, lab_dl2, lab_dl3, unl_dl0, unl_dl1, unl_dl2, unl_dl3):
            xl, yl = lab_bat
#             xo0, yo0 = lab0_bat
            xo1, yo1 = lab1_bat
            xo2, yo2 = lab2_bat
            xo3, yo3 = lab3_bat

            xu0, yu0 = unl0_bat
            xu1, yu1 = unl1_bat
            xu2, yu2 = unl2_bat
            xu3, yu3 = unl3_bat

            start_time = time.time()

            pl, _ = model(xl.cuda())
#             _, po0 = model(xo0.cuda())
            _, po1 = model(xo1.cuda())
            _, po2 = model(xo2.cuda())
            _, po3 = model(xo3.cuda())

            _, pu0 = model(xu0.cuda())
            _, pu1 = model(xu1.cuda())
            _, pu2 = model(xu2.cuda())
            _, pu3 = model(xu3.cuda())

            loss = _loss([pl, po1, po2, po3, pu0, pu1, pu2, pu3],
                         [yl.cuda(), yo1.cuda(), yo2.cuda(), yo3.cuda(), yu0.cuda(), yu1.cuda(), yu2.cuda(), yu3.cuda()],
                         train_opts['w'], train_opts['label_original'])
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()
            step += 1

            duration += time.time() - start_time
            if step % nsteps_show_loss == 0:
                samples_per_s = batch_size * nsteps_show_loss / duration
                duration = 0.0
                print('=> %s: step %d, loss = %f, lr = %f (%.1f samples/sec)' % (
                    datetime.now(), step, loss.item(), scheduler.get_last_lr()[0], samples_per_s))

            if step % nsteps_check != 0:
                continue

            losses = np.append(losses, loss.item())
            np.save(losses_file, losses)

            print('=> Evaluating on validation dataset...')
            accuracy, _ = evaluator.evaluate(model)
            print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

            if accuracy > best_accuracy:
                print('=> Model saved to file: %s' % model.store(log_dir, step=step))
                patience = init_patience
                best_accuracy = accuracy
            else:
                patience -= 1

            print('=> patience = %d' % patience)
            if patience == 0:
                return


def main(args):
    path_to_log_dir = args.logdir
    if not os.path.exists(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    training_options = {
        'logdir': args.logdir,
        'restore_checkpoint': args.restore_checkpoint,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate,
        'label_size': args.label_size,
        'label_unlabel_ratio': args.label_unlabel_ratio,
        'best_acc': args.best_acc,
        'w': args.w,
        'val_test': args.val_test,
        'label_original': args.label_original,
        'wideresnet': args.wideresnet
    }
    print(training_options)

    print('Start training')
    _train(training_options)
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
