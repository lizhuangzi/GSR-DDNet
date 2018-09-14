import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('normT', nn.BatchNorm2d(num_input_features))
        self.add_module('reluT', nn.ReLU(inplace=True))
        self.add_module('convT', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class SRDDNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6,12,24,16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True):

        super(SRDDNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('firstConv', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('prelu0', nn.PReLU())

        num_features = num_init_features
        self.block1 = _DenseBlock(num_layers=block_config[0],
                            num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate,
                            drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate

        self.trans1 = _Transition(num_input_features=num_features,
                            num_output_features=int(num_features * compression))

        num_features = int(num_features * compression)
        INP = num_features + num_init_features
        self.block2 = _DenseBlock(num_layers=block_config[1],
                                  num_input_features=INP,
                                  bn_size=bn_size, growth_rate=growth_rate,
                                  drop_rate=drop_rate)
        num_features = INP + block_config[1] * growth_rate

        self.trans2 = _Transition(num_input_features=num_features,
                                  num_output_features=int(num_features * compression))
        num_features = int(num_features * compression)
        INP = num_features + INP
        self.block3 = _DenseBlock(num_layers=block_config[2],
                                  num_input_features=INP,
                                  bn_size=bn_size, growth_rate=growth_rate,
                                  drop_rate=drop_rate)
        num_features = INP + block_config[2] * growth_rate

        self.trans3 = _Transition(num_input_features=num_features,
                                  num_output_features=int(num_features * compression))

        num_features = int(num_features * compression)
        INP = num_features + INP
        self.block4 = _DenseBlock(num_layers=block_config[3],
                                  num_input_features=INP,
                                  bn_size=bn_size, growth_rate=growth_rate,
                                  drop_rate=drop_rate)
        num_features = INP + block_config[3] * growth_rate
        self.trans4 = _Transition(num_input_features=num_features,
                                  num_output_features=int(num_features * compression))


        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(2450),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2450, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        )


        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU()
        )
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)



    def forward(self, x):

        features = self.features(x)
        b1 = self.block1(features)
        t1 = self.trans1(b1)
        concat = torch.cat([features, t1], 1)

        b2 = self.block2(concat)
        t2 = self.trans2(b2)
        concat = torch.cat([concat, t2], 1)

        b3 = self.block3(concat)
        t3 = self.trans3(b3)
        concat = torch.cat([concat, t3], 1)

        b4 = self.block4(concat)
        t4 = self.trans4(b4)
        concat = torch.cat([concat, b4], 1)

        bottleneck = self.bottleneck(concat)
        de = self.deconv(bottleneck)
        out = self.reconstruction(de)
        return out
