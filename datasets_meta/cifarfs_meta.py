import numpy as np
import os
import json
import h5py
from PIL import Image
from torchmeta.datasets.utils import get_asset

from torchvision.datasets.utils import check_integrity, download_url
# from torchmeta.utils.data import Dataset, ClassDataset
from task_meta import Dataset
from dataset_meta import ClassDataset, CombinationMetaDataset


class CIFAR100ClassDataset(ClassDataset):
    folder = 'cifar100'
    subfolder = None
    download_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    gz_folder = 'cifar-100-python'
    gz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    files_md5 = {
        'train': '16019d7e3df5f24257cddd939b257f8d',
        'test': 'f0ef6b0ae62326f3e7ffdfab6717acfc',
        'meta': '7973b15100ade9c7d40fb424638fde48'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}_labels.json'
    filename_fine_names = 'fine_names.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CIFAR100ClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                                   meta_split=meta_split,
                                                   class_augmentations=class_augmentations)

        if self.subfolder is None:
            raise ValueError()

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self._data = None
        self._labels = None

        self.split_filename_labels = os.path.join(self.root, self.subfolder,
                                                  self.filename_labels.format(self.meta_split))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('CIFAR100 integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        coarse_label_name, fine_label_name = self.labels[index % self.num_classes]
        data = self.data['{0}/{1}'.format(coarse_label_name, fine_label_name)]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CIFAR100Dataset(index, data,
                               coarse_label_name, fine_label_name,
                               transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(os.path.join(self.root, self.filename), 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (self._check_integrity_data()
            and os.path.isfile(self.split_filename_labels)
            and os.path.isfile(os.path.join(self.root, self.filename_fine_names)))

    def _check_integrity_data(self):
        return os.path.isfile(os.path.join(self.root, self.filename))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import tarfile
        import pickle
        import shutil

        if self._check_integrity_data():
            return

        gz_filename = '{0}.tar.gz'.format(self.gz_folder)
        download_url(self.download_url, self.root, filename=gz_filename,
                     md5=self.gz_md5)
        with tarfile.open(os.path.join(self.root, gz_filename), 'r:gz') as tar:
            tar.extractall(path=self.root)

        train_filename = os.path.join(self.root, self.gz_folder, 'train')
        check_integrity(train_filename, self.files_md5['train'])
        with open(train_filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            images = data[b'data']
            fine_labels = data[b'fine_labels']
            coarse_labels = data[b'coarse_labels']

        test_filename = os.path.join(self.root, self.gz_folder, 'test')
        check_integrity(test_filename, self.files_md5['test'])
        with open(test_filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            images = np.concatenate((images, data[b'data']), axis=0)
            fine_labels = np.concatenate((fine_labels, data[b'fine_labels']), axis=0)
            coarse_labels = np.concatenate((coarse_labels, data[b'coarse_labels']), axis=0)

        images = images.reshape((-1, 3, 32, 32))
        images = images.transpose((0, 2, 3, 1))

        meta_filename = os.path.join(self.root, self.gz_folder, 'meta')
        check_integrity(meta_filename, self.files_md5['meta'])
        with open(meta_filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            fine_label_names = data['fine_label_names']
            coarse_label_names = data['coarse_label_names']

        filename = os.path.join(self.root, self.filename)
        fine_names = dict()
        with h5py.File(filename, 'w') as f:
            for i, coarse_name in enumerate(coarse_label_names):
                group = f.create_group(coarse_name)
                fine_indices = np.unique(fine_labels[coarse_labels == i])
                for j in fine_indices:
                    dataset = group.create_dataset(fine_label_names[j], data=images[fine_labels == j])
                fine_names[coarse_name] = [fine_label_names[j] for j in fine_indices]

        filename_fine_names = os.path.join(self.root, self.filename_fine_names)
        with open(filename_fine_names, 'w') as f:
            json.dump(fine_names, f)

        gz_folder = os.path.join(self.root, self.gz_folder)
        if os.path.isdir(gz_folder):
            shutil.rmtree(gz_folder)
        if os.path.isfile('{0}.tar.gz'.format(gz_folder)):
            os.remove('{0}.tar.gz'.format(gz_folder))


class CIFAR100Dataset(Dataset):
    def __init__(self, index, data, coarse_label_name, fine_label_name,
                 transform=None, target_transform=None):
        super(CIFAR100Dataset, self).__init__(index, transform=transform,
                                              target_transform=target_transform)
        self.data = data
        self.coarse_label_name = coarse_label_name
        self.fine_label_name = fine_label_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = (self.coarse_label_name, self.fine_label_name)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)


class CIFARFS(CombinationMetaDataset):
    """
    The CIFAR-FS dataset, introduced in [1]. This dataset contains
    images of 100 different classes from the CIFAR100 dataset [2].
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `cifar100` exists.
    num_classes_per_task : int
        Number of classes per tasks. This corresponds to `N` in `N-way`
        classification.
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.
    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `transforms.ClassSplitter()`.
    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g. `transforms.HorizontalFlip()`.
    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `cifar100` folder). If the dataset is already
        available, this does not download/process the dataset again.
    Notes
    -----
    The meta train/validation/test splits are over 64/16/20 classes from the
    CIFAR100 dataset.
    References
    ----------
    .. [1] Bertinetto L., Henriques J. F., Torr P. H.S., Vedaldi A. (2019).
           Meta-learning with differentiable closed-form solvers. In International
           Conference on Learning Representations (https://arxiv.org/abs/1805.08136)
    .. [2] Krizhevsky A. (2009). Learning Multiple Layers of Features from Tiny
           Images. (https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
    """

    def __init__(self, root, task_generate_method=None, num_classes_per_task=None, num_classes_distractor=None,
                 meta_train=False, meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CIFARFSClassDataset(root, meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                      meta_split=meta_split,
                                      transform=transform, class_augmentations=class_augmentations,
                                      download=download)
        super(CIFARFS, self).__init__(dataset, task_generate_method, num_classes_per_task,
                                      num_classes_distractor=num_classes_distractor,
                                      target_transform=target_transform, dataset_transform=dataset_transform)


class CIFARFSClassDataset(CIFAR100ClassDataset):
    subfolder = 'cifar-fs'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CIFARFSClassDataset, self).__init__(root, meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                                  meta_split=meta_split,
                                                  transform=transform,
                                                  class_augmentations=class_augmentations,
                                                  download=download)

    def download(self):
        if self._check_integrity():
            return
        super(CIFARFSClassDataset, self).download()

        subfolder = os.path.join(self.root, self.subfolder)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        for split in ['train', 'val', 'test']:
            split_filename_labels = os.path.join(subfolder, self.filename_labels.format(split))
            if os.path.isfile(split_filename_labels):
                continue

            data = get_asset(self.folder, self.subfolder, '{0}.json'.format(split), dtype='json')
            with open(split_filename_labels, 'w') as f:
                json.dump(data, f)


if __name__ == '__main__':
    from datasets_meta.splitters_meta import ClassSplitter, ClassSplitterDist
    from torchmeta.transforms import Categorical
    from torchvision.transforms import ToTensor, Resize, Compose

    name = 'cifarfs'
    folder = '/home/cxl173430/data/DATASETS/'  # do not change the folder address

    # transform = Compose([Resize(32), ToTensor()])
    transform = Compose([ToTensor()])
    dataset = CIFARFSClassDataset(folder, meta_train=True, meta_val=False, meta_test=False,
                                  meta_split=None, transform=transform, class_augmentations=None,
                                  download=True)

    print(dataset[0])
    '''
    dataset[0][0][0].shape
    Out[3]: torch.Size([3, 32, 32])
    dataset[0][0][1]
    Out[4]: (('aquatic_mammals', 'dolphin'), None)

    len(dataset)
    Out[1]: 64                   
    len(dataset[0])
    Out[2]: 600

    '''

    # ##### 1. this part is to test the regular ClassSplitter, "woDistractor"
    num_ways = 5
    num_shots = 2
    num_shots_test = 15
    num_shots_unlabel = 3
    hidden_size = 64
    num_classes_distractor = 0

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabeled_per_class=num_shots_unlabel)

    meta_train_dataset = CIFARFS(folder,
                                 task_generate_method='woDistractor',
                                 transform=transform,
                                 target_transform=Categorical(num_ways),
                                 num_classes_per_task=num_ways,
                                 num_classes_distractor=num_classes_distractor,
                                 meta_train=True,
                                 dataset_transform=dataset_transform,
                                 download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_1 = meta_train_dataset[(0, 1, 3, 4, 5)]
    print("total number of samples in the task:", len(task_1['train']) + len(task_1['test']) + len(task_1['unlabeled']))
    # total number of samples in the task: 63 #5*2 + 5*15 + 5*3= 10+75+15=100
    print('image shape:', task_1['train'][0][0].shape)  # image shape: torch.Size([3, 32, 32])
    print('image label:', task_1['train'][0][1])  # image label: 2
    pass

    # ##### END test for 1

    ##### 2. this part is to test  "Distractor"
    num_ways = 5
    num_shots = 2
    num_shots_test_meta_train = 15
    num_shots_distractor = 2  # since the num of classes is 64, this number should be > 64
    num_classes_distractor = 3

    dataset_transform2 = ClassSplitterDist(shuffle=True,
                                          num_train_per_class=num_shots,
                                          num_test_per_class=num_shots_test_meta_train,
                                          num_unlabeled_per_class=num_shots_unlabel,
                                          num_unlabel_OOD_per_class=num_shots_distractor)

    meta_train_dataset = CIFARFS(folder,
                                 task_generate_method='distractor',
                                 transform=transform,
                                 target_transform=Categorical(num_ways),
                                 num_classes_per_task=num_ways,
                                 num_classes_distractor=num_classes_distractor,
                                 meta_train=True,
                                 dataset_transform=dataset_transform2,
                                 download=True)


    # sample_task = meta_train_dataset.sample_task()
    task_2 = meta_train_dataset[(0, 1, 3, 5, 2, 7, 8, 9)]
    print("total number of samples in the task:", len(task_2['train']) + len(task_2['test']) + len(task_2['unlabeled']))
    # 5*2=10, 5*15 = 75, 5*3 + 3*2 = 21
    print('image shape:', task_2['train'][0][0].shape)
    print('image label:', task_2['train'][0][1])
    pass

    '''
    total number of samples in the task: 106
    image shape: torch.Size([3, 32, 32])
    image label: 3
    '''
    ##### END test for 2