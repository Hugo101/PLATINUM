import numpy as np
import torch
from torchvision.datasets import SVHN
from torchvision import transforms

from task_meta import Dataset
from dataset_meta import ClassDataset, CombinationMetaDataset


class SVHN_meta(CombinationMetaDataset):
    def __init__(self, root,
                 task_generate_method=None,
                 num_classes_per_task=None,
                 meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None,
                 download=False):
        dataset = SVHN_ClassDataset(root,
                                    meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                    meta_split=meta_split,
                                    transform=transform,
                                    class_augmentations=class_augmentations,
                                    download=download)
        super(SVHN_meta, self).__init__(dataset, task_generate_method, num_classes_per_task,
                                        target_transform=target_transform,
                                        dataset_transform=dataset_transform)


class SVHN_ClassDataset(ClassDataset):
    def __init__(self, root,
                 meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None,
                 transform=None,
                 class_augmentations=None,
                 download=False):
        super(SVHN_ClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                                meta_split=meta_split,
                                                class_augmentations=class_augmentations)

        self.root = root
        self.transform = transform
        self.data = SVHN(self.root, split="train", download=download, transform=self.transform)

        if meta_train:
            self._labels = ['2', '3', '7', '4', '0', '5']
        if meta_val or meta_test:
            self._labels = ['1', '6', '8', '9']

        self._num_classes = len(self.labels)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        return self._labels


    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        idx_original = self.data.labels == int(class_name)
        data_one_class = torch.utils.data.Subset(self.data, np.where(idx_original == 1)[0])
        target_transform = self.get_target_transform(index)

        return SVHN_one_class(index, data_one_class, class_name, target_transform)


class SVHN_one_class(Dataset):
    def __init__(self, index, data, class_name, target_transform):
        super(SVHN_one_class, self).__init__(index)
        self.data = data     # torch.utils.data.Subset
        self.class_name = class_name

    def __len__(self):
        # return self.data.shape[0]
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        target = self.class_name
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)


if __name__ == '__main__':
    from splitters_meta import ClassSplitter
    from torchmeta.transforms import Categorical
    from torchvision.transforms import ToTensor, Resize, Compose

    # name = 'mnist'
    folder = '/home/cxl173430/data/'
    num_ways = 3
    num_shots = 2
    num_shots_test = 15
    num_shots_unlabel = 4
    # hidden_size = 64

    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,)),
                                          # lambda x: x.view(1, 28, 28),
                                          ])
    #input to the MNIST dataset

    #this works well
    dataset = SVHN_ClassDataset(folder, meta_train=True, meta_val=False, meta_test=False, meta_split=None,
                                       transform=transformations, class_augmentations=None,
                                       download=True)

    print(dataset[0])
    '''
    dataset[0][0][0].shape 
    Out[3]: torch.Size([3, 32, 32])
    dataset[0][0][1]
    Out[4]: '2'
    '''

    hidden_size = 64
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabel_per_class=num_shots_unlabel)

    meta_train_dataset = SVHN_meta(folder,
                                      transform=transformations,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_1 = meta_train_dataset[(0, 1, 2)]
    print('image shape:', task_1['train'][0][0].shape)
    print('image label:', task_1['train'][0][1])
    pass

    '''
    task1['train'][0][0].shape 
    Out[11]: torch.Size([3, 32, 32])
    task1['train'][0][1] 
    Out[12]: 2
    '''