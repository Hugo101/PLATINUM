import torch
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision import transforms
import numpy as np

transform = Compose([ToTensor(), transforms.Normalize((0.5,), (0.5,))])
root = "/home/cxl173430/data/DATASETS/"
svhn_data_train = SVHN(root, split="train", download=True, transform=transform)


for label in range(10):
    count = np.sum(svhn_data_train.labels == label)
    print(f"The number of samples for the class {label}: {count}")

'''
svhn_data_train: (ndarray: 73257,3,32,32)

The number of samples for the class 0: 4948
The number of samples for the class 1: 13861
The number of samples for the class 2: 10585
The number of samples for the class 3: 8497
The number of samples for the class 4: 7458
The number of samples for the class 5: 6882
The number of samples for the class 6: 5727
The number of samples for the class 7: 5595
The number of samples for the class 8: 5045
The number of samples for the class 9: 4659

meta-train: ['2', '3', '7', '4', '0', '5']
10585, 8497, 5595, 7458, 4948, 6882

meta-test: ['1', '6', '8', '9']
13861, 5727, 5045, 4659
'''