import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from task_meta import Dataset
from dataset_meta import ClassDataset, CombinationMetaDataset


class MNIST_meta(CombinationMetaDataset):
    def __init__(self, root, task_generate_method=None, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MNIST_ClassDataset(root, meta_train=meta_train,
                                           meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                           transform=transform, class_augmentations=class_augmentations,
                                           download=download)
        super(MNIST_meta, self).__init__(dataset, task_generate_method, num_classes_per_task,
                                           target_transform=target_transform, dataset_transform=dataset_transform)
class MNIST_ClassDataset(ClassDataset):
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MNIST_ClassDataset, self).__init__(meta_train=meta_train,
                                                       meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                       class_augmentations=class_augmentations)

        self.root = root
        self.transform = transform

        self.data = MNIST(self.root, train=True, download=download, transform=self.transform)
        self.data_no_label = torch.unsqueeze(self.data.data, 1).float()

        if meta_train:
            # randomly select 6 classes as meta-train: 2,3,7,4,0,5
            # idx = mnist.targets == 2
            # idx += mnist.targets == 3
            # idx += mnist.targets == 7
            # idx += mnist.targets == 4
            # idx += mnist.targets == 0
            # idx += mnist.targets == 5
            #
            # data_sub = torch.utils.data.Subset(mnist, np.where(idx == 1)[0])

            # self._data = None #TODO: modify
            self._labels = ['2', '3', '7', '4', '0', '5']
            # self.num_classes = len(self.labels)

        if meta_val or meta_test:
            # idx_t = mnist.targets == 1
            # idx_t += mnist.targets == 6
            # idx_t += mnist.targets == 8
            # idx_t += mnist.targets == 9
            #
            # data_sub = torch.utils.data.Subset(mnist, np.where(idx_t == 1)[0])

            # self._data = None #TODO: modify
            self._labels = ['1', '6', '8', '9']
            # self.num_classes = len(self.labels)

        self._num_classes = len(self.labels)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        return self._labels


    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        idx_original = self.data.targets == int(class_name)
        data_one_class = self.data_no_label[idx_original]
        target_transform = self.get_target_transform(index)

        return MNIST_one_class(index, data_one_class, class_name, target_transform)

class MNIST_one_class(Dataset):
    def __init__(self, index, data, class_name, target_transform): #TODO, attention, this target_transform actually helps
        super(MNIST_one_class, self).__init__(index)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data[index]
        target = self.class_name
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)


if __name__ == '__main__':
    from splitters_meta import ClassSplitter, ClassSplitterComUnlabel
    from torchmeta.transforms import Categorical
    from torchvision.transforms import ToTensor, Resize, Compose

    # name = 'mnist'
    folder = '/home/cxl173430/data/ssl_maml_dataset/'

    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,)),
                                          lambda x: x.view(1, 28, 28),
                                          ])
    #input to the MNIST dataset

    #this works well
    dataset = MNIST_ClassDataset(folder, meta_train=True, meta_val=False, meta_test=False, meta_split=None,
                                       transform=transformations, class_augmentations=None,
                                       download=True)

    '''
    dataset[0][0][0].shape
    Out[3]: torch.Size([1, 28, 28])
    dataset[0][0][1]
    Out[4]: '2'

    len(dataset)
    Out[1]: 6                   '2', '3', '7', '4', '0', '5'
    len(dataset[0])
    Out[2]: 5958
    len(dataset[1])
    Out[3]: 6131
    len(dataset[2])
    Out[4]: 6265
    len(dataset[3])
    Out[5]: 5842
    len(dataset[4])
    Out[6]: 5923
    len(dataset[5])
    Out[7]: 5421
    '''

    # ##### 1. this part is to test the regular ClassSplitter, "woDistractor"
    num_ways = 3
    num_shots = 2
    num_shots_test = 15
    num_shots_unlabel = 4
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabel_per_class=num_shots_unlabel)

    meta_train_dataset = MNIST_meta(folder,
                                    task_generate_method='woDistractor',
                                    transform=transformations,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_1 = meta_train_dataset[(0, 1, 3)]
    print("total number of samples in the task:", len(task_1['train'])+len(task_1['test'])+len(task_1['unlabeled']))
    # total number of samples in the task: 63 #3*2 + 3*15 + 3*4= 6+45+12=63
    print('image shape:', task_1['train'][0][0].shape) # image shape: torch.Size([1, 28, 28])
    print('image label:', task_1['train'][0][1]) #image label: 1
    pass


    # ##### END test for 1

    ##### 2. this part is to test  "random"
    num_ways = 3
    num_shots = 2
    num_shots_test = 15
    num_unlabel_total = 40
    dataset_transform2 = ClassSplitterComUnlabel(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabel_total=num_unlabel_total)

    meta_train_dataset = MNIST_meta(folder,
                                    task_generate_method='random',
                                    transform=transformations,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform2,
                                    download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_2 = meta_train_dataset[(0, 1, 3)]
    print("total number of samples in the task:", len(task_2['train'])+len(task_2['test'])+len(task_2['unlabeled']))
    print('image shape:', task_2['train'][0][0].shape)
    print('image label:', task_2['train'][0][1])
    pass

    '''
    task1['train'][0][0].shape 
    Out[11]: torch.Size([1, 28, 28])
    task1['train'][0][1] 
    Out[12]: 0
    '''
    ##### END test for 2