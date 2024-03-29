__package__ = 'MAML_with_SSL.maml'
import torch
import torch.nn as nn
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
import copy
from maml.resnet import resnet12

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None, last=False, freeze=False, keep_feat=False): #modified for SMI function
        if freeze:
            with torch.no_grad():
                features = self.features(inputs, params=self.get_subdict(params, 'features')) # if params is None, get_subdict returns None
                features = features.view((features.size(0), -1))

        else:
            features = self.features(inputs, params=self.get_subdict(params, 'features'))
            features_flatten = features.view((features.size(0), -1))

        logits = self.classifier(features_flatten, params=self.get_subdict(params, 'classifier'))

        # if last:
        #     return logits, features_flatten
        if keep_feat:
            return logits, features
        else:
            return logits

    def get_embedding_dim(self): #added for SMI function
        self.embDim = self.feature_size
        return self.embDim


    def update_batch_stats(self, flag): #added for PL, not used actually
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

def ModelConvCIFARFS(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=2 * 2 * hidden_size)

def ModelConvSVHN(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=2 * 2 * hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)


class MetaResNet12Model(MetaModule):
    def __init__(self, out_features, hidden_size=64, feature_size=640):
        super(MetaResNet12Model, self).__init__()
        # self.in_channels = in_channels
        self.out_features = out_features
        # self.hidden_size = hidden_size
        self.feature_size = feature_size

        # pretrained model
        model_path = "/data/cxl173430/MAML_SMI_sefDefine/maml/few-shot-models/mini_simple.pth"
        self.features = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        self.features.load_state_dict(ckpt['model'])

        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None, last=False, freeze=True, keep_feat=False): #modified for SMI function
        params_classifier = params

        if not freeze and params:
            params_classifier = OrderedDict()
            params_feat = {}
            #  params_copy = copy.deepcopy(params)
            params_copy = params.copy()
            for k,v in params_copy.items():
                if "classifier" in k:
                    params_classifier[k] = params_copy[k]
                else:
                    params_feat[k] = params_copy[k]

            model_dict = self.features.state_dict()
            model_dict.update(params_feat)
            self.features.load_state_dict(model_dict)

        if freeze:
            with torch.no_grad():
                features, _ = self.features(inputs, is_feat=True) # if params is None, get_subdict returns None
                features_flatten = features[-1].view((features[-1].size(0), -1))

        else:
            features, _ = self.features(inputs, is_feat=True)
            features_flatten = features[-1].view((features[-1].size(0), -1))

        logits = self.classifier(features_flatten, params=self.get_subdict(params_classifier, 'classifier'))

        # if last:
        #     return logits, features_flatten
        if keep_feat:
            return logits, features
        else:
            return logits

    def get_embedding_dim(self): #added for SMI function
        self.embDim = self.feature_size
        return self.embDim


    def update_batch_stats(self, flag): #added for PL, not used actually
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


if __name__ == '__main__':

    from torchinfo import summary

    model = ModelConvMiniImagenet(1, hidden_size=64)

    batch_size = 4

    summary = summary(model, input_size=(batch_size, 3, 32, 32))
    print(summary)
