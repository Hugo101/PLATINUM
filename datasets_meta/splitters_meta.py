import torch
import numpy as np
import random
from collections import OrderedDict, defaultdict
# from torchmeta.utils.data.task import Task, ConcatTask, SubsetTask
from task_meta import Task, ConcatTask, SubsetTask, SubsetCombine, SubsetTask_unlabel
from torchmeta.transforms.utils import apply_wrapper

from configuration import arg_parser
args = arg_parser.parse_args()

__all__ = ['Splitter', 'ClassSplitter', 'ClassSplitterDist', 'ClassSplitterComUnlabel']

RATIO_LABELED = args.ratio
print("~~~~~~~~~~~~~~~~~~ Splitting ratio of labeled samples:", RATIO_LABELED)

class Splitter(object):
    def __init__(self, splits, random_state_seed):
        self.splits = splits      # OrderedDict([('train', 2), ('test',15), ('unlabeled', 4)])
        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def get_indices(self, task):
        if isinstance(task, ConcatTask):
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError('The task must be of type `ConcatTask` or `Task`, Got type `{0}`.'.format(type(task)))
        return indices

    def get_indices_task(self, task):
        raise NotImplementedError('Method `get_indices_task` must be '
            'implemented in classes inherited from `Splitter`.')

    def get_indices_concattask(self, task):
        raise NotImplementedError('Method `get_indices_concattask` must be '
            'implemented in classes inherited from `Splitter`.')

    def _get_class_indices(self, task):
        class_indices = defaultdict(list)
        if task.num_classes is None: # Regression task
            class_indices['regression'] = range(len(task))
        else:
            for index in range(len(task)):
                sample = task[index]
                if (not isinstance(sample, tuple)) or (len(sample) < 2):
                    raise ValueError('In order to split the dataset in train/'
                        'test splits, `Splitter` must access the targets. Each '
                        'sample from a task must be a tuple with at least 2 '
                        'elements, with the last one being the target.')
                class_indices[sample[-1]].append(index)

            if len(class_indices) != task.num_classes:
                raise ValueError('The number of classes detected in `Splitter` '
                    '({0}) is different from the property `num_classes` ({1}) '
                    'in task `{2}`.'.format(len(class_indices),
                    task.num_classes, task))

        return class_indices

    def __call__(self, task):
        indices = self.get_indices(task)
        # #### todo: note, used for debug
        # print(f"^^^^^^^^^^ The selected sample indices for this task in support set: {indices['train']}")
        # print(f"^^^^^^^^^^ The selected sample indices for this task in query set: {indices['test']}")
        # print(f"^^^^^^^^^^ The selected sample indices for this task in unlabeled set: {indices['unlabeled']}")
        # ####
        return OrderedDict([(split, SubsetTask(task, indices[split])) for split in self.splits])

    def __len__(self):
        return len(self.splits)


class ClassSplitter_(Splitter):
    def __init__(self, shuffle=True, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None, num_unlabeled_per_class=None,
                 random_state_seed=0):
        """
        Transforms a dataset into train/test splits for few-shot learning tasks,
        based on a fixed number of samples per class for each split. This is a
        dataset transformation to be applied as a `dataset_transform` in a
        `MetaDataset`.

        Parameters
        ----------
        shuffle : bool (default: `True`)
            Shuffle the data in the dataset before the split.

        num_samples_per_class : dict, optional
            Dictionary containing the names of the splits (as keys) and the
            corresponding number of samples per class in each split (as values).
            If not `None`, then the arguments `num_train_per_class`,
            `num_test_per_class`, `num_support_per_class` and
            `num_query_per_class` are ignored.

        num_train_per_class : int, optional
            Number of samples per class in the training split. This corresponds
            to the number of "shots" in "k-shot learning". If not `None`, this
            creates an item `train` for each task.

        num_test_per_class : int, optional
            Number of samples per class in the test split. If not `None`, this
            creates an item `test` for each task.

        random_state_seed : int, optional
            seed of the np.RandomState. Defaults to '0'.

        Examples
        """
        self.shuffle = shuffle

        if num_samples_per_class is None:
            num_samples_per_class = OrderedDict()
            if num_train_per_class is not None:
                num_samples_per_class['train'] = num_train_per_class
            if num_test_per_class is not None:
                num_samples_per_class['test'] = num_test_per_class
            if num_unlabeled_per_class is not None:                          #new added
                num_samples_per_class['unlabeled'] = num_unlabeled_per_class   #new added

        assert len(num_samples_per_class) > 0

        self._min_samples_per_class = sum(num_samples_per_class.values())
        super(ClassSplitter_, self).__init__(num_samples_per_class, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        for i, (name, class_indices) in enumerate(all_class_indices.items()):
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for class `{0}` ({1}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({2}).'.format(name,
                    num_samples, self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + i + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0
        for dataset in task.datasets:           # task.datasets is the samples of selected classes
            num_samples_total = len(dataset)    # total num of samples for one class, i.e., 600 for miniimagenet

            # num of labeled samples, i.e., 600*0.4=240, the rest are unlabeled
            num_samples_labeled = int(num_samples_total * RATIO_LABELED)
            if num_samples_total < self._min_samples_per_class:
                raise ValueError('The number of samples for one class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({1}).'.format(num_samples_total,
                    self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples_labeled)    # 240
                dataset_indices_unlabeled = np.random.RandomState(seed).permutation(
                    num_samples_total - num_samples_labeled) + num_samples_labeled    # 360
            else:
                dataset_indices = np.arange(num_samples_labeled)
                dataset_indices_unlabeled = np.arange(num_samples_labeled, num_samples_total, 1)

            ptr = 0
            for split, num_split in self.splits.items():
                if split != 'unlabeled':
                    split_indices = dataset_indices[ptr:ptr + num_split]
                    if self.shuffle:
                        self.np_random.shuffle(split_indices)
                    indices[split].extend(split_indices + cum_size)
                    ptr += num_split
                else:
                    if num_split > 0:   # the num of unlabeled samples is nonzero
                        split_indices_unlabeled = dataset_indices_unlabeled[0 : num_split]
                        if self.shuffle:
                            self.np_random.shuffle(split_indices_unlabeled)
                        indices[split].extend(split_indices_unlabeled + cum_size)

            cum_size += num_samples_total
        return indices

# def ClassSplitter(task=None, *args, **kwargs):
#     return apply_wrapper(ClassSplitter_(*args, **kwargs), task)

def ClassSplitter(*args, **kwargs):
    return ClassSplitter_(*args, **kwargs)


class ClassSplitterDist(object):
    def __init__(self, shuffle=True, splits=None,
                 num_train_per_class=None, num_test_per_class=None, num_unlabeled_per_class=None,
                 num_unlabel_OOD_per_class=None,
                 random_state_seed=0):
        '''
        dataset transform to generate task with distractor classes
        '''
        self.splits = splits
        self.num_distractors_per_class = num_unlabel_OOD_per_class
        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)

        self.shuffle = shuffle
        if splits is None:
            self.splits = OrderedDict()
            if num_train_per_class is not None:
                self.splits['train'] = num_train_per_class
            if num_test_per_class is not None:
                self.splits['test'] = num_test_per_class
            if num_unlabeled_per_class is not None:
                self.splits['unlabeled'] = num_unlabeled_per_class


        assert len(self.splits) > 0
        self._min_samples_per_class = sum(self.splits.values())

    def __call__(self, task_ID, task_distractor):
        '''
        to sample a task with distractor classes
        :param task_ID: the concatenation of all examples of selected ID classes
        :param task_distractor: the concatenation of all examples of selected OOD classes
        :return: desired task
        '''
        indices = self.get_indices(task_ID)
        indices_unlabeled_OOD = self.get_indices_concattask_OOD(task_distractor)

        # ##### todo: note, used for debug
        # print("^^^^^^^^^^ The selected classes indices: {}".format(index))
        # print("^^^^^^^^^^ The selected sample indices for this task in support set: {}".format(indices['train']))
        # print("^^^^^^^^^^ The selected sample indices for this task in query set: {}".format(indices['test']))
        # print("^^^^^^^^^^ The selected sample indices for this task in unlabeled set: {}".format(indices['unlabeled']))

        # #####
        support_query_unlabel = OrderedDict()
        for split in self.splits:
            if "unlabeled" != split:  # support set and query set
                support_query_unlabel[split] = SubsetTask(task_ID, indices[split])
            else: # unlabeled set (ID classes, and distractor classes)
                support_query_unlabel[split] = SubsetCombine(task_ID, indices[split], task_distractor, indices_unlabeled_OOD)

        return support_query_unlabel

    def get_indices(self, task):
        if isinstance(task, ConcatTask):
            indices = self.get_indices_concattask(task)
        else:
            raise ValueError("The task must be of type `ConcatTask`, Got type `{0}`.".format(type(task)))
        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0
        for dataset in task.datasets:  # task.datasets is the samples of selected classes
            num_samples_total = len(dataset)  # total num of samples for one class, i.e., 600 for miniimagenet

            # num of labeled samples, i.e., 600*0.4=240, the rest are unlabeled
            num_samples_labeled = int(num_samples_total * RATIO_LABELED)
            if num_samples_total < self._min_samples_per_class:
                raise ValueError('The number of samples for one class ({0}) '
                                 'is smaller than the minimum number of samples per class '
                                 'required by `ClassSplitter` ({1}).'.format(num_samples_total,
                                                                             self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples_labeled)  # 240
                dataset_indices_unlabeled = np.random.RandomState(seed).permutation(
                    num_samples_total - num_samples_labeled) + num_samples_labeled  # 360
            else:
                dataset_indices = np.arange(num_samples_labeled)
                dataset_indices_unlabeled = np.arange(num_samples_labeled, num_samples_total, 1)

            ptr = 0
            for split, num_split in self.splits.items():
                if split != 'unlabeled':
                    split_indices = dataset_indices[ptr:ptr + num_split]
                    if self.shuffle:
                        self.np_random.shuffle(split_indices)
                    indices[split].extend(split_indices + cum_size)
                    ptr += num_split
                else:
                    if num_split > 0:  # the num of unlabeled samples is nonzero
                        split_indices_unlabeled = dataset_indices_unlabeled[0: num_split]
                        if self.shuffle:
                            self.np_random.shuffle(split_indices_unlabeled)
                        indices[split].extend(split_indices_unlabeled + cum_size)

            cum_size += num_samples_total
        return indices


    def get_indices_concattask_OOD(self, task):
        '''
        get the indices for distractor examples in the unlabeled set
        :param task: the concatenation of all samples of selected classes (distractor)
        :return: indices
        '''
        indices = []
        cum_size = 0
        # 1. find the potential indices
        for dataset in task.datasets:               # task.datasets is the selected classes
            num_samples_total = len(dataset)        # total num of samples for one class, i.e., 600 for miniimagenet
            num_samples_labeled = int(num_samples_total * RATIO_LABELED) # num of labeled samples, i.e., 600*0.4=240

            if self.shuffle:
                seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices_unlabeled = np.random.RandomState(seed).permutation(
                    np.arange(num_samples_labeled, num_samples_total, 1))
            else:
                dataset_indices_unlabeled = np.arange(num_samples_labeled, num_samples_total, 1)

            # for each class, select unlabeled distractor examples
            split_indices_unlabeled = dataset_indices_unlabeled[0: self.num_distractors_per_class]
            if self.shuffle:
                self.np_random.shuffle(split_indices_unlabeled)
            indices.extend(split_indices_unlabeled+cum_size)

            cum_size += num_samples_total # jump to next class
        return indices

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)


# the other splitting methods for unlabeled samples.
# 1) all unlabeled samples are a set. For each task, we select some unlabeled samples from this total unlabeled set.
class ClassSplitterComUnlabel(object):
    def __init__(self, shuffle=True, splits=None,
                 num_train_per_class=None, num_test_per_class=None, num_unlabeled_total=None,
                 random_state_seed=0):
        '''
        Transforms a dataset into support/query/unlabeled splits for semi supervised few-shot learning tasks,
        based on a fixed number of samples per class for split "support" (train) and "query" (test) and the total number
        of unlabeled samples for this task.
        This is a dataset transformation to be applied as a `dataset_transform` in a `MetaDataset`.

        Parameters
        ------------
        :param shuffle: bool (default: `True`)
        :param splits: : dict. num_samples_per_class for "train"(support), "test"(query), and whole unlabeled set
        :param num_train_per_class:
        :param num_test_per_class:
        :param num_unlabel_total:
        :param random_state_seed:
        '''
        self.splits = splits
        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)

        self.shuffle = shuffle
        if splits is None:
            self.splits = OrderedDict()
            if num_train_per_class is not None:
                self.splits['train'] = num_train_per_class

            if num_test_per_class is not None:
                self.splits['test'] = num_test_per_class

            if num_unlabeled_total is not None:
                self.splits['unlabeled'] = num_unlabeled_total

        assert len(self.splits) > 0

    def __call__(self, task, dataset_class, index):
        '''
        to sample a task
        :param task: the concatenation of all samples of selected classes
        :param dataset_class: the transformed dataset, for example, MNIST_ClassDataset
        :return: desired task
        '''
        indices, ratio_ID_class, indices_OOD = self.get_indices(task, dataset_class, index)

        # ##### todo: note, used for debug
        print("^^^^^^^^^^ The selected classes indices: {}".format(index))
        print("^^^^^^^^^^ The selected sample indices for this task in support set: {}".format(indices['train']))
        print("\n********** The ratio of ID samples in the selected unlabeled set:{:.4f}".format(ratio_ID_class))
        # #####
        # support_query = OrderedDict([(split, SubsetTask(task, indices[split])) for split in self.splits])
        support_query_unlabel = OrderedDict()
        for split in self.splits:
            if "unlabel" in split:
                # ####
                # #v1: cannot log the label of OOD and ID info
                # support_query_unlabel[split] = SubsetTask(dataset_class, indices[split])
                # #v2: improve v1
                support_query_unlabel[split] = SubsetTask_unlabel(dataset_class, indices[split], indices_OOD)
                # ##### method 2: works for mnist and SVHN
                # support_query_unlabel[split] = SubsetUnlabel(dataset_class, indices[split])
            else:
                support_query_unlabel[split] = SubsetTask(task, indices[split])

        return support_query_unlabel

    def get_indices(self, task, dataset_class, index):
        if isinstance(task, ConcatTask):
            # ################# random selection TODO: make the code more clear
            indices, ratio_ID_class, indices_OOD = self.get_indices_concattask(task, dataset_class, index)
            # ################# All OOD selection
            # indices, ratio_ID_class = self.get_indices_concattask_all_OOD(task, dataset_class)
        else:
            raise ValueError("The task must be of type `ConcatTask`, "
                             "Got type `{0}`.".format(type(task)))
        return indices, ratio_ID_class, indices_OOD


    def get_indices_concattask(self, task, dataset_class, index):
        '''
        get the indices for support set, query set, and unlabeled set
        :param task: the concatenation of all samples of selected classes
        :param dataset_class: the transformed dataset, for example, MNIST_ClassDataset
        :index:
        :return: indices
        '''
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0
        # 1. labeled part: support set and query set
        for dataset in task.datasets:               # task.datasets is the selected classes
            num_samples_total = len(dataset)        # total num of samples for one class, i.e., 600 for miniimagenet

            # num of labeled samples, i.e., 600*0.4=240, the rest are unlabeled
            num_samples_labeled = int(num_samples_total * RATIO_LABELED)

            if self.shuffle:
                seed = (hash(task) + hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices_labeled = np.random.RandomState(seed).permutation(num_samples_labeled)  # 240 if 0.4*600
            else:
                dataset_indices_labeled = np.arange(num_samples_labeled)
            # for each class, select support examples and query examples
            ptr = 0
            for split, num_split in self.splits.items(): # train: 5, test: 15, unlabeled: 100
                if "unlabel" in split:
                    continue
                else:
                    split_indices = dataset_indices_labeled[ptr : ptr+num_split]
                    if self.shuffle:
                        self.np_random.shuffle(split_indices)
                    indices[split].extend(split_indices+cum_size)
                    ptr += num_split
            cum_size += num_samples_total # jump to next class

        # 2. unlabeled part
        # 1) correct version
        key_unlabel = 'unlabeled'
        # #######
        total_unlabel_list = []
        total_unlabel_ID_list = []
        cum_size_unlabel = 0
        for i in range(len(dataset_class.datasets)):  # total number of classes
            dataset = dataset_class.datasets[i]
            num_samples_total = len(dataset)
            num_samples_labeled = int(num_samples_total * RATIO_LABELED)
            if self.shuffle:
                seed = (hash(dataset) + self.random_state_seed) % (2 ** 32)
                dataset_indices_unlabeled = np.random.RandomState(seed).permutation(
                    num_samples_total - num_samples_labeled) + num_samples_labeled
            else:
                dataset_indices_unlabeled = np.arange(num_samples_labeled, num_samples_total, 1)
            total_unlabel_list.extend((dataset_indices_unlabeled + cum_size_unlabel))
            if i in index:
                total_unlabel_ID_list.extend(dataset_indices_unlabeled + cum_size_unlabel)

            cum_size_unlabel += num_samples_total

        # randomly select the unlabeled set from the pool of total unlabeled samples
        unlabel_selected = random.sample(total_unlabel_list, self.splits[key_unlabel])
        indices[key_unlabel] = unlabel_selected

        # calculate the common element, namely the number of samples of ID classes.
        common_num = len(set(total_unlabel_ID_list).intersection(set(unlabel_selected)))

        indices_OOD = list(set(unlabel_selected) - set(total_unlabel_ID_list))

        return indices, common_num/self.splits[key_unlabel], indices_OOD


    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

# def SubsetUnlabel(dataset_class, index):
#     '''
#     :param dataset_class: the transformed dataset, for example, MNIST_ClassDataset
#     :param indices: list of list
#     :return: selected unlabeled set
#     '''
#     from torch.utils.data import ConcatDataset, Subset
#     selected_unlabel = []
#     for (idx, data) in enumerate(dataset_class):
#         tmp_data_one_class = Subset(data, index[idx])
#         tmp_data_one_class.dataset.class_name = 1 #label is not meaningful
#         selected_unlabel.append(tmp_data_one_class)
#
#     selected_unlabel_samples = ConcatDataset(selected_unlabel)
#
#     # labels = torch.ones(1, len(selected_unlabel_samples))
#     # selected_unlabel_samples[1] = labels
#     return selected_unlabel_samples
