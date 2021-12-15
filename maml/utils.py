import torch
from collections import OrderedDict
from torchmeta.modules import MetaModule
from torch.utils.data import Dataset
import torchvision.transforms as T

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DatasetSMI(Dataset):
    def __init__(self, img, target):
        self.img = img
        self.target = target

    def __getitem__(self, index):
        x = self.img[index]
        y = self.target[index]
        return (x,y)

    def __len__(self):
        return len(self.target)


class DatasetAugment(Dataset):
    def __init__(self, img, target):
        self.img = img
        self.target = target
        # self.transform = T.RandAugment() # torchvision 0.11.0
        self.transform = T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)
        self.num = len(self.target)*20

    def __getitem__(self, index):
        local_idx = index % len(self.target)
        x = self.img[local_idx]
        x = self.transform(x)
        y = self.target[local_idx]
        return (x,y)

    def __len__(self):
        return self.num



# model updates in the inner loop
'''
model parameters (old):
for i,j in self.model.named_parameters():
    print(i, j.shape)
    
features.layer1.conv.weight torch.Size([64, 3, 3, 3])
features.layer1.conv.bias torch.Size([64])
features.layer1.norm.weight torch.Size([64])            #norm, not included in updated params
features.layer1.norm.bias torch.Size([64])              #norm, not included in updated params
features.layer2.conv.weight torch.Size([64, 64, 3, 3])
features.layer2.conv.bias torch.Size([64])
features.layer2.norm.weight torch.Size([64])            #norm, not included in updated params
features.layer2.norm.bias torch.Size([64])              #norm, not included in updated params
features.layer3.conv.weight torch.Size([64, 64, 3, 3])
features.layer3.conv.bias torch.Size([64])
features.layer3.norm.weight torch.Size([64])            #norm, not included in updated params
features.layer3.norm.bias torch.Size([64])              #norm, not included in updated params
features.layer4.conv.weight torch.Size([64, 64, 3, 3])
features.layer4.conv.bias torch.Size([64])
features.layer4.norm.weight torch.Size([64])            #norm, not included in updated params
features.layer4.norm.bias torch.Size([64])              #norm, not included in updated params
classifier.weight torch.Size([3, 256])
classifier.bias torch.Size([3])

params (new):
for k,v in params.items():
    print(k, v.shape)
    
features.layer1.conv.weight torch.Size([64, 3, 3, 3])
features.layer1.conv.bias torch.Size([64])
features.layer2.conv.weight torch.Size([64, 64, 3, 3])
features.layer2.conv.bias torch.Size([64])
features.layer3.conv.weight torch.Size([64, 64, 3, 3])
features.layer3.conv.bias torch.Size([64])
features.layer4.conv.weight torch.Size([64, 64, 3, 3])
features.layer4.conv.bias torch.Size([64])
classifier.weight torch.Size([3, 256])
classifier.bias torch.Size([3])
'''

def model_update(model, params):
    if params is None:
        return model
    # read the parameters of model, which is a collections.OrderedDict
    model_dict = model.state_dict()
    # update the acquired model parameters based on params, which is the updated parameters
    model_dict.update(params)
    # load the updated params to model
    model.load_state_dict(model_dict)
    return model