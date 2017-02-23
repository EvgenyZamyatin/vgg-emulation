import os
from argparse import ArgumentParser
from time import time

import h5py
import numpy as np
import sys
from scipy.misc import imread, imsave, imresize

from src.optimize import optimize

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/vgg19.pkl'
TRAIN_PATH = 'data/train2014.h5'
BATCH_SIZE = 4


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--size', type=int,
                        dest='size',
                        help='images size',
                        default=256)
    parser.add_argument('--checkpoint-model', type=str,
                        dest='checkpoint_model',
                        help='path to checkpoint model',
                        default=False)

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists(options.checkpoint_dir):
        os.makedirs(options.checkpoint_dir)
    with open('command.txt', 'w+') as out:
        out.write(' '.join(sys.argv))

    size = (options.size, options.size)

    db = h5py.File(options.train_path, 'r')
    content_targets = db['train2014']['images']

    kwargs = {
        "epochs": options.epochs,
        "period": options.checkpoint_iterations,
        "batch_size": options.batch_size,
        "save_path": os.path.join(options.checkpoint_dir, 'fns.ckpt'),
        "learning_rate": options.learning_rate,
    }

    if options.checkpoint_model is not False:
        kwargs["checkpoint_model"] = options.checkpoint_model

    args = [
        content_targets,
        options.vgg_path,
        size,
    ]

    last_time = time()
    for epoch, i, loss in optimize(*args, **kwargs):

        print('Epoch %d, Iteration: %d, Loss: %1.3lf, Time: %1.3lf' % (epoch, i, loss, time() - last_time))
        last_time = time()

if __name__ == '__main__':
    main()