from time import time

import theano
import theano.tensor as T
import numpy as np
from src import emul, vgg
from lasagne.updates import adam
import os
import pickle


def l2_loss(x, y):
    return ((x - y) ** 2).sum()


def save(file, obj):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w+b') as out:
        pickle.dump(obj, out)


def optimize(content_targets, vgg_path, im_size,
             epochs=2, period=1000,
             batch_size=4, save_path='saver/fns.ckpt',
             learning_rate=1e-3, checkpoint_model=None):

    assert content_targets.shape[1:] == (3, *im_size)

    print('=== CREATE VGG NET ===')
    images = T.ftensor4()
    vgg_net = vgg.Net(im_size)
    vgg_net.set_params(vgg.load_params(vgg_path))

    print('=== CREATE EMUL NET ===')
    emul_net = emul.Net(im_size)
    if checkpoint_model is not None:
        emul_net.set_params(checkpoint_model)

    content_losses = []
    for layer in vgg.CONTENT_LAYERS + vgg.STYLE_LAYERS:
        content_vgg_features = vgg_net(images, layer)
        content_emul_features = emul_net(images, layer)
        size = content_emul_features.size
        loss = l2_loss(content_vgg_features, content_emul_features) / size
        content_losses.append(loss)

    loss = sum(content_losses)
    updates = adam(loss, emul_net.get_params(False), learning_rate)
    print('=== FUNCTION COMPILE ===')
    train_fn = theano.function([images], loss, updates=updates)
    valid_fn = theano.function([images], loss)
    print('=== START TRAIN ===')
    it = 0
    time_in_train = 0
    for epoch in range(epochs):
        for i in range(0, len(content_targets), batch_size):
            batch = content_targets[i:i + batch_size]
            batch = np.float32(batch)
            start = time()
            loss = train_fn(batch)
            time_in_train += time() - start
            it += 1
            if it % period == 0 or i + batch_size >= len(content_targets):
                print('Time in train: %1.3lf' % time_in_train)
                time_in_train = 0
                save(save_path, emul_net.get_params())
                yield epoch, it, loss