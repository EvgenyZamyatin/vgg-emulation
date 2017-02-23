import numpy as np
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, set_all_param_values, Layer, get_all_param_values, \
    get_all_params
from lasagne.layers import get_output
import pickle
import theano.tensor as T


class Net:
    def __init__(self, size):
        net = {}
        net['input'] = InputLayer((None, 3, *size))

        net['conv1_1'] = Conv2DLayer(net['input'], 64, 3, pad=1)
        net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)

        net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)

        net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1)
        net['pool3'] = MaxPool2DLayer(net['conv3_2'], 2)

        net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1)
        net['pool4'] = MaxPool2DLayer(net['conv4_2'], 2)

        net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1)
        self.net = net

    def __call__(self, images, layer):
        return get_output(self.net[layer], images)

    def set_params(self, params):
        if isinstance(params, str):
            with open(params, 'rb') as data:
                params = pickle.load(data)
        set_all_param_values(self.net['conv5_1'], params)

    def get_params(self, values=True):
        if values:
            return get_all_param_values(self.net['conv5_1'])
        return get_all_params(self.net['conv5_1'], trainable=True)