import torch
import torch.nn as nn
import torch.nn.functional as F

class swn(nn.Module):
    def __init__(self, channels, stride=1, isCIFARFS=False):
        super(swn, self).__init__()

        self.in_channels = 128
        self.activation = nn.LeakyReLU()
        self.isCIFARFS = isCIFARFS

        layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        # conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(self.activation)

        conv_layer.append(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(1))

        layers.append(nn.Sequential(*conv_layer))

        self.layers = layers

        if not isCIFARFS:
            self.linear_1 = nn.Linear(25, 8)
            self.linear_2 = nn.Linear(8, 1)
        else:
            self.linear_1 = nn.Linear(4, 1)


    def forward(self, input):
        out = self.layers[0](input)

        out = nn.Flatten()(out)
        out = self.activation(out)
        out = self.linear_1(out)

        if self.isCIFARFS:
            return out

        out = self.linear_2(out)
        return out



