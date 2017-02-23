import numpy as np
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.layers import set_all_param_values, get_output
import pickle

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
CONTENT_LAYERS = ('conv4_2',)


MEAN_PIXEL = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 3, 1, 1))


def load_params(file):
    with open(file, 'rb') as data:
        params = pickle.load(data, encoding='latin1')
    return params['param values'][:32]


class Net:
    def __init__(self, size):
        net = {}
        net['input'] = InputLayer((None, 3, *size))
        net['conv1_1'] = Conv2DLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
        net['conv2_1'] = Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
        net['conv3_1'] = Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_4'] = Conv2DLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = MaxPool2DLayer(net['conv3_4'], 2)
        net['conv4_1'] = Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_4'] = Conv2DLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = MaxPool2DLayer(net['conv4_4'], 2)
        net['conv5_1'] = Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['conv5_4'] = Conv2DLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = MaxPool2DLayer(net['conv5_4'], 2)
        self.net = net

    def __call__(self, images, layer):
        images_pre = images - MEAN_PIXEL
        images_pre = images_pre[:, ::-1]
        return get_output(self.net[layer], images_pre)

    def set_params(self, params):
        set_all_param_values(self.net['pool5'], params)