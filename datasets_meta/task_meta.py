from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import Dataset as Dataset_
from torchvision.transforms import Compose

__all__ = ['Dataset', 'Task', 'ConcatTask', 'SubsetTask']


class Dataset(Dataset_):
    def __init__(self, index, transform=None, target_transform=None):
        self.index = index
        self.transform = transform
        self.target_transform = target_transform

    def target_transform_append(self, transform):
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])

    def __hash__(self):
        return hash(self.index)


class Task(Dataset):
    """Base class for a classification task.

    Parameters
    ----------
    num_classes : int
        Number of classes for the classification task.
    """
    def __init__(self, index, num_classes,
                 transform=None, target_transform=None):
        super(Task, self).__init__(index, transform=transform,
                                   target_transform=target_transform)
        self.num_classes = num_classes


class ConcatTask(Task, ConcatDataset):
    def __init__(self, datasets, num_classes, target_transform=None):
        index = tuple(task.index for task in datasets)
        Task.__init__(self, index, num_classes)
        ConcatDataset.__init__(self, datasets)
        for task in self.datasets: #self.datasets is from ConcatDataset: list(datasets)
            task.target_transform_append(target_transform)

    def __getitem__(self, index):
        return ConcatDataset.__getitem__(self, index)


class SubsetTask(Task, Subset):
    def __init__(self, dataset, indices, num_classes=None,
                 target_transform=None):
        if num_classes is None:
            num_classes = dataset.num_classes
        Task.__init__(self, dataset.index, num_classes)
        Subset.__init__(self, dataset, indices)
        self.dataset.target_transform_append(target_transform)

    def __getitem__(self, index):
        return Subset.__getitem__(self, index)

    def __hash__(self):
        return hash((self.index, tuple(self.indices)))


# class SubsetUnlabel(Subset):
#     def __init__(self, dataset, indices, indices_ood):
#         Subset.__init__(self, dataset, indices)
#         self.indices_ood = indices_ood
#
#     def __getitem__(self, index):
#         img, label = Subset.__getitem__(self, index)
#

class SubsetTask_unlabel(Task, Subset):
    def __init__(self, dataset, indices, indices_ood, num_classes=None,
                 target_transform=None):
        if num_classes is None:
            num_classes = dataset.num_classes
        Task.__init__(self, dataset.index, num_classes)
        Subset.__init__(self, dataset, indices)
        self.dataset.target_transform_append(target_transform)

        self.indices_ood = indices_ood

    def __getitem__(self, index):
        # return Subset.__getitem__(self, index)

        index_pick = self.indices[index]
        img, label = self.dataset[index_pick]
        if index_pick not in self.indices_ood:
            return img, label #ID sample
        else:
            return img, -1   #OOD sample

    def __hash__(self):
        return hash((self.index, tuple(self.indices)))




