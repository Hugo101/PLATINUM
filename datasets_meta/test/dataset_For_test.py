import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
import sys
sys.path.append("../../")
from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)

if __name__ == "__main__":
    # name = 'miniimagenet'
    # folder = '/home/cxl173430/data/miniimagenet_test'
    # num_ways = 5
    # num_shots = 2
    # num_shots_test = 15
    # num_shots_unlabel = 3
    # hidden_size = 64

    # transform = Compose([Resize(84), ToTensor()])
    # meta_train_dataset = MiniImagenet(folder,
    #                                   transform=transform,
    #                                   target_transform=Categorical(num_ways),
    #                                   num_classes_per_task=num_ways,
    #                                   meta_train=True,
    #                                   dataset_transform=dataset_transform,
    #                                   download=True)
    #
    # # sample_task = meta_train_dataset.sample_task()
    # task_1 = meta_train_dataset[(0,1,2,3,4)]
    # # task_2 = meta_train_dataset[(0, 1, 2, 3, 66)]
    # '''
    # task_1['train'][0][0].shape
    # Out[3]: torch.Size([3, 84, 84])
    # task_1['train'][0][1]
    # Out[4]: 0
    # '''
    num_ways = 5
    num_shots = 2
    num_shots_test = 15
    folder_o = '/home/cxl173430/data'
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    transform_o = Compose([Resize(28), ToTensor()])
    meta_train_dataset = Omniglot(folder_o,
                                    transform=transform_o,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)
    task_2 = meta_train_dataset[(0, 1, 2, 3, 4)]
    print('image shape:', task_2['train'][0][0].shape)
    print('image label:', task_2['train'][0][1])

    '''
    task_2['train'][0][0].shape 
    Out[2]: torch.Size([1, 28, 28])
    task_2['train'][0][1]
    Out[3]: 3
    '''

    pass