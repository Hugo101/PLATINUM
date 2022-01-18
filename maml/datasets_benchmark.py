import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision import transforms
from collections import namedtuple
from torchmeta.transforms import Categorical, Rotation

from datasets_meta.omniglot_meta import Omniglot
from datasets_meta.miniimagenet_meta import MiniImagenet
from datasets_meta.mnist_meta import MNIST_meta
from datasets_meta.SVHN_meta import SVHN_meta
from datasets_meta.tieredimagenet_meta import TieredImagenet
from datasets_meta.cifarfs_meta import CIFARFS
from datasets_meta.splitters_meta import ClassSplitter, ClassSplitterDist, ClassSplitterComUnlabel

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelConvSVHN, ModelConvCIFARFS
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')


def get_benchmark_by_name(name,
                          folder,
                          task_generate_method,
                          num_ways,
                          num_shots,
                          num_shots_test_meta_train,
                          num_shots_test_meta_test,
                          num_shots_unlabel,            # with and without distractor
                          num_shots_unlabel_eval,       # with and without distractor
                          num_classes_distractor,       # with distractor
                          num_shots_distractor,         # with distractor
                          num_shots_distractor_eval,    # with distractor
                          num_unlabeled_total,
                          num_unlabeled_total_evaluate,
                          hidden_size=None):
    # several task generating methods: "woDistractor"(proof of concept), "random" (baseline)
    dataset_transform, dataset_transform_evaluate = None, None
    if task_generate_method == "woDistractor":
        assert num_shots_distractor == 0 and num_classes_distractor == 0 and num_shots_distractor_eval == 0
        dataset_transform = ClassSplitter(shuffle=True,
                                          num_train_per_class=num_shots,
                                          num_test_per_class=num_shots_test_meta_train,
                                          num_unlabeled_per_class=num_shots_unlabel)

        dataset_transform_evaluate = ClassSplitter(shuffle=True,                 # make it to be false to debug
                                                   num_train_per_class=num_shots,
                                                   num_test_per_class=num_shots_test_meta_test,
                                                   num_unlabeled_per_class=num_shots_unlabel_eval)

    elif task_generate_method == "distractor":
        assert num_classes_distractor > 0 and num_shots_distractor > 0 and num_shots_distractor_eval > 0
        dataset_transform = ClassSplitterDist(shuffle=True,
                                              num_train_per_class=num_shots,
                                              num_test_per_class=num_shots_test_meta_train,
                                              num_unlabeled_per_class=num_shots_unlabel,
                                              num_unlabel_OOD_per_class=num_shots_distractor)

        dataset_transform_evaluate = ClassSplitterDist(shuffle=True,                 # make it to be false to debug
                                                       num_train_per_class=num_shots,
                                                       num_test_per_class=num_shots_test_meta_test,
                                                       num_unlabeled_per_class=num_shots_unlabel_eval,
                                                       num_unlabel_OOD_per_class=num_shots_distractor_eval)

    elif task_generate_method == "random":
        dataset_transform = ClassSplitterComUnlabel(shuffle=True,
                                                    num_train_per_class=num_shots,
                                                    num_test_per_class=num_shots_test_meta_train,
                                                    num_unlabeled_total=num_unlabeled_total)

        dataset_transform_evaluate = ClassSplitterComUnlabel(shuffle=True,       # make it to be false to debug
                                                             num_train_per_class=num_shots,
                                                             num_test_per_class=num_shots_test_meta_test,
                                                             num_unlabeled_total=num_unlabeled_total_evaluate)

    if name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          task_generate_method,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          num_classes_distractor=num_classes_distractor,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset   = MiniImagenet(folder,
                                          task_generate_method,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          num_classes_distractor=num_classes_distractor,
                                          meta_val=True,
                                          dataset_transform=dataset_transform_evaluate)
        meta_test_dataset  = MiniImagenet(folder,
                                          task_generate_method,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          num_classes_distractor=num_classes_distractor,
                                          meta_test=True,
                                          dataset_transform=dataset_transform_evaluate)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'tieredimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = TieredImagenet(folder,
                                            task_generate_method,
                                            transform=transform,
                                            target_transform=Categorical(num_ways),
                                            num_classes_per_task=num_ways,
                                            num_classes_distractor=num_classes_distractor,
                                            meta_train=True,
                                            dataset_transform=dataset_transform,
                                            download=True)
        meta_val_dataset   = TieredImagenet(folder,
                                            task_generate_method,
                                            transform=transform,
                                            target_transform=Categorical(num_ways),
                                            num_classes_per_task=num_ways,
                                            num_classes_distractor=num_classes_distractor,
                                            meta_val=True,
                                            dataset_transform=dataset_transform_evaluate)
        meta_test_dataset  = TieredImagenet(folder,
                                            task_generate_method,
                                            transform=transform,
                                            target_transform=Categorical(num_ways),
                                            num_classes_per_task=num_ways,
                                            num_classes_distractor=num_classes_distractor,
                                            meta_test=True,
                                            dataset_transform=dataset_transform_evaluate)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size) # todo: check
        loss_function = F.cross_entropy

    elif name == 'cifarfs':
        transform = Compose([ToTensor()])
        meta_train_dataset = CIFARFS(folder,
                                     task_generate_method,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     num_classes_distractor=num_classes_distractor,
                                     meta_train=True,
                                     dataset_transform=dataset_transform,
                                     download=True)

        meta_val_dataset   = CIFARFS(folder,
                                     task_generate_method,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     num_classes_distractor=num_classes_distractor,
                                     meta_val=True,
                                     dataset_transform=dataset_transform_evaluate)

        meta_test_dataset   = CIFARFS(folder,
                                      task_generate_method,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      num_classes_distractor=num_classes_distractor,
                                      meta_test=True,
                                      dataset_transform=dataset_transform_evaluate)
        model = ModelConvCIFARFS(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy


    elif name == 'omniglot': # todo: modify further
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      task_generate_method,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset   = Omniglot(folder,
                                      task_generate_method,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_val=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform)
        meta_test_dataset  = Omniglot(folder,
                                      task_generate_method,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_test=True,
                                      dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'mnist':
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = MNIST_meta(folder,
                                        task_generate_method,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_train=True,
                                        dataset_transform=dataset_transform,
                                        download=True)
        meta_val_dataset = MNIST_meta(folder,
                                      task_generate_method,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_val=True,
                                      dataset_transform=dataset_transform_evaluate)
        meta_test_dataset = MNIST_meta(folder,
                                       task_generate_method,
                                       transform=transform,
                                       target_transform=Categorical(num_ways),
                                       num_classes_per_task=num_ways,
                                       meta_test=True,
                                       dataset_transform=dataset_transform_evaluate)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'svhn':
        transform = Compose([ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        meta_train_dataset = SVHN_meta(folder,
                                       task_generate_method,
                                       transform=transform,
                                       target_transform=Categorical(num_ways),
                                       num_classes_per_task=num_ways,
                                       meta_train=True,
                                       dataset_transform=dataset_transform,
                                       download=True)
        meta_val_dataset = SVHN_meta(folder,
                                     task_generate_method,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_val=True,
                                     dataset_transform=dataset_transform_evaluate)
        meta_test_dataset = SVHN_meta(folder,
                                      task_generate_method,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_test=True,
                                      dataset_transform=dataset_transform_evaluate)

        model = ModelConvSVHN(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)


if __name__ == "__main__":
    from torchvision.transforms import ToTensor, Resize, Compose
    from torchvision import transforms
    # name = 'miniimagenet'
    folder = '/home/cxl173430/data/miniimagenet_test'
    num_ways = 5
    num_shots = 2
    num_shots_test = 15
    num_shots_unlabel = 3
    hidden_size = 64
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabel_per_class=num_shots_unlabel)
    transform = Compose([Resize(84), ToTensor()])
    meta_train_dataset = MiniImagenet(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_1 = meta_train_dataset[(0, 1, 2, 3, 4)]
    # task_2 = meta_train_dataset[(0, 1, 2, 3, 66)]
    '''
    task_1['train'][0][0].shape 
    Out[3]: torch.Size([3, 84, 84])
    task_1['train'][0][1] 
    Out[4]: 0
    '''

    # name = 'mnist'
    folder = '/home/cxl173430/data/'
    num_ways = 3
    num_shots = 2
    num_shots_test = 15
    num_shots_unlabel = 4
    # hidden_size = 64

    transformations = Compose([ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                               lambda x: x.view(1, 28, 28),
                               ])
    # input to the MNIST dataset

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test,
                                      num_unlabel_per_class=num_shots_unlabel)

    meta_train_dataset = MNIST_meta(folder,
                                    transform=transformations,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)

    # sample_task = meta_train_dataset.sample_task()
    task_2 = meta_train_dataset[(0, 1, 2)]
    print('image shape:', task_2['train'][0][0].shape)
    print('image label:', task_2['train'][0][1])

    '''
    task_2['train'][0][0].shape 
    Out[3]: torch.Size([1, 28, 28])
    task_2['train'][0][1] 
    Out[4]: 0
    '''

    pass
