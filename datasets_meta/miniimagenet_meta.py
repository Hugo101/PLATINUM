import os
import pickle
from PIL import Image
import h5py
import json
# from torchmeta.utils.data import Dataset
from task_meta import Dataset
from dataset_meta import ClassDataset, CombinationMetaDataset
# from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
# QKFIX: See torchmeta.datasets_self.utils for more informations
from torchmeta.datasets.utils import download_file_from_google_drive


class MiniImagenet(CombinationMetaDataset):
    """
    The Mini-Imagenet dataset, introduced in [1]. This dataset contains images
    of 100 different classes from the ILSVRC-12 dataset (Imagenet challenge).
    The meta train/validation/test splits are taken from [2] for reproducibility.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `miniimagenet` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
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
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `miniimagenet` folder). If the dataset is already
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The meta train/
    validation/test splits are over 64/16/20 classes.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)

    .. [2] Ravi, S. and Larochelle, H. (2016). Optimization as a Model for
           Few-Shot Learning. (https://openreview.net/forum?id=rJY0-Kcll)
    """

    def __init__(self, root, task_generate_method=None, num_classes_per_task=None, num_classes_distractor=None,
                 meta_train=False, meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MiniImagenetClassDataset(root, meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                           meta_split=meta_split,
                                           transform=transform, class_augmentations=class_augmentations,
                                           download=download)
        super(MiniImagenet, self).__init__(dataset, task_generate_method, num_classes_per_task,
                                           num_classes_distractor=num_classes_distractor,
                                           target_transform=target_transform, dataset_transform=dataset_transform)


class MiniImagenetClassDataset(ClassDataset):
    folder = 'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MiniImagenetClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                                       meta_split=meta_split,
                                                       class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
                                           self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
                                                  self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return MiniImagenetDataset(index, data, class_name,
                                   transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets_self']   # mark the dataset
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
                                        self.gz_filename, md5=self.gz_md5)

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image_data'], data['class_dict']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets_self')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)


class MiniImagenetDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(MiniImagenetDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)


if __name__ == '__main__':
    from splitters_meta import ClassSplitter, ClassSplitterComUnlabel
    from torchmeta.transforms import Categorical
    from torchvision.transforms import ToTensor, Resize, Compose
    name = 'miniimagenet'
    folder = '/home/cxl173430/data/DATASETS/miniimagenet_test_test' #do not change the folder address

    transform = Compose([Resize(84), ToTensor()])
    dataset = MiniImagenetClassDataset(folder, meta_train=True, meta_val=False, meta_test=False, meta_split=None,
                                       transform=transform, class_augmentations=None,
                                       download=True)

    print(dataset[0])
    '''
    dataset[0][0][0].shape
    Out[3]: torch.Size([3, 84, 84])
    dataset[0][0][1]
    Out[4]: ('n01532829', None)

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

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabel_per_class=num_shots_unlabel)

    meta_train_dataset = MiniImagenet(folder,
                                    task_generate_method='woDistractor',
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_1 = meta_train_dataset[(0, 1, 3, 4, 5)]
    print("total number of samples in the task:", len(task_1['train']) + len(task_1['test']) + len(task_1['unlabeled']))
    # total number of samples in the task: 63 #3*2 + 3*15 + 3*4= 6+45+12=63
    print('image shape:', task_1['train'][0][0].shape)  # image shape: torch.Size([1, 28, 28])
    print('image label:', task_1['train'][0][1])  # image label: 1
    pass

    # ##### END test for 1

    ##### 2. this part is to test  "random"
    num_ways = 3
    num_shots = 2
    num_shots_test = 15
    num_unlabel_total = 200 #since the num of classes is 64, this number should be > 64
    dataset_transform2 = ClassSplitterComUnlabel(shuffle=True,
                                                 num_train_per_class=num_shots,
                                                 num_test_per_class=num_shots_test,
                                                 num_unlabel_total=num_unlabel_total)

    meta_train_dataset = MiniImagenet(folder,
                                    task_generate_method='random',
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform2,
                                    download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_2 = meta_train_dataset[(0, 1, 3)]
    print("total number of samples in the task:", len(task_2['train']) + len(task_2['test']) + len(task_2['unlabeled']))
    print('image shape:', task_2['train'][0][0].shape)
    print('image label:', task_2['train'][0][1])
    pass

    '''
    total number of samples in the task: 251
    image shape: torch.Size([3, 84, 84])
    image label: 1
    '''
    ##### END test for 2