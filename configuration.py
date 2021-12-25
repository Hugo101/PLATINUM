import argparse

arg_parser = argparse.ArgumentParser('MAML + SSL (SMI, PL, VAT and FixMatch(later)) ')

# General
arg_parser.add_argument('--seed', type=int, default=123)
arg_parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU available. index of gpu, if <0 then use cpu')
arg_parser.add_argument('--data_folder', type=str, default='/home/cxl173430/data/DATASETS/miniimagenet_test',
                        help='Path to the folder the data is downloaded to.')
arg_parser.add_argument('--output_folder', type=str, default="output/",
                        help='Path to the output folder to save the model.')
arg_parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['mnist', 'omniglot', 'miniimagenet', 'svhn'],
                        help='Name of the dataset (default: omniglot).')
arg_parser.add_argument('--ratio', type=float, default=0.4,
                        help='ratio of labeled for each class in the task.')


# SSL Experimental Setting
arg_parser.add_argument('--scenario', type=str, default="woDistractor",
                        choices=["woDistractor", "distractor", "random" , "allOOD"],
                        help="Different SS FSL approaches, including subset selection and baselines")
arg_parser.add_argument('--ssl_algo', type=str, default='SMI')  # "PL", "VAT", "SMI", "PLtopZ"
arg_parser.add_argument('--selection_option', type=str, default='cross')  # "same", "cross", "union"
arg_parser.add_argument('--type_smi', type=str, default='vanilla')  # "vanilla", "rank", "gain"
arg_parser.add_argument('--ssl_algo_meta_test', type=str, default='mamlTestLargeS')  # "no",  "mamlTestLargeS"
arg_parser.add_argument('--coef_inner', type=float, default=-1,
                        help='coefficient of ssl loss function in the inner loop.')
arg_parser.add_argument('--coef_outer', type=float, default=-1,
                        help='coefficient of ssl loss function in the outer loop.')


# SMI
arg_parser.add_argument("--sf", type=str, default="fl2mi")
arg_parser.add_argument("--budget_s", type=int, default=25)  # 30 for 1-shot, 50 for 5-shot, for support set
arg_parser.add_argument("--budget_q", type=int, default=75)  # for query set


arg_parser.add_argument('--num_ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way").')
arg_parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of examples per class for support set (k in "k-shot").')
arg_parser.add_argument('--num_shots_test', type=int, default=15,
                        help='Number of examples per class for query set. If negative, same as `--num_shots`(default:15).')

# unlabeled part, for with and without distractor
arg_parser.add_argument('--num_shots_unlabeled', type=int, default=50,
                        help='Number of unlabeled example per class.')  # 200 for SVHN, 300 for MNIST, 20 for miniImagenet
arg_parser.add_argument('--num_shots_unlabeled_evaluate', type=int, default=50,
                        help='Number of unlabeled example per class during meta-validation/test.')  # 200 for SVHN, 300 for MNIST, 20 for miniImagenet
# unlabeled part, Scenario: distractor.
arg_parser.add_argument('--num_classes_distractor', type=int, default=0,
                        help='Number of distractor classes.')
arg_parser.add_argument('--num_shots_distractor', type=int, default=0,
                        help='Number of unlabeled example per distractor class during meta-training.')
arg_parser.add_argument('--num_shots_distractor_eval', type=int, default=0,
                        help='Number of unlabeled example per distractor class during meta-validation/test.')


# unlabeled part, for random selection
arg_parser.add_argument('--num_unlabel_total', type=int, default=100,
                        help='Num of unlabeled examples totally for each task. (default: 600 for SVHN, 900 for MNIST).')
arg_parser.add_argument('--num_unlabel_total_evaluate', type=int, default=100,
                        help='Number of unlabeled examples totally for each task during meta-validation/test.')

# CNN Model
arg_parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of channels in each convolution layer of the VGG network (default: 64).')

# Optimization
arg_parser.add_argument('--first_order', action='store_true',
                        help='Use the first order approximation, do not use higher-order derivatives during meta-optimization.')
arg_parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of tasks in a batch of tasks for meta-training.')
arg_parser.add_argument('--batch_size_val', type=int, default=1,
                        help='Number of tasks in a batch of tasks for meta-validation.')
arg_parser.add_argument('--batch_size_test', type=int, default=1,     # todo: check this later
                        help='Number of tasks in a batch of tasks for meta-test.')


arg_parser.add_argument('--num_epochs', type=int, default=400,
                        help='Number of epochs of meta-training (default: 50).')
arg_parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batch of tasks per epoch (default: 100).')
arg_parser.add_argument('--step_size', type=float, default=0.001,
                        help='Size of the fast adaptation step, ie. learning rate in the '
                             'gradient descent update (default: 0.1).')
arg_parser.add_argument('--meta_lr', type=float, default=0.0001,
                        help='Learning rate for the meta-optimizer (optimization of the outer '
                             'loss). The default optimizer is Adam (default: 1e-3).')
arg_parser.add_argument('--num_steps', type=int, default=5,
                        help='Number of fast adaptation steps, ie. gradient descent updates (default: 1).')
arg_parser.add_argument('--num_steps_evaluate', type=int, default=10,
                        help='Number of fast adaptation steps in valid/test, ie. gradient descent updates.')


# PL with threshold or top Z
arg_parser.add_argument('--pl_threshold', type=float, default=0,
                        help='The threshold used in the PL algorithm in the inner loop. (Default: 0)')
arg_parser.add_argument('--pl_threshold_outer', type=float, default=0,
                        help='The threshold used in the PL algorithm in the outer loop. (Default: 0)')
arg_parser.add_argument("--pl_num_topz", type=int, default=25,
                        help='The number of examples which have top Z probability logits in the inner loop')
arg_parser.add_argument("--pl_num_topz_outer", type=int, default=75,
                        help='The number of examples which have top Z probability logits in the outer loop')
arg_parser.add_argument("--pl_batch_size", type=int, default=80)


# Miscellaneous
arg_parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers to use for data-loading (default: 1).')
arg_parser.add_argument('--verbose', action='store_true')

# debugging purpose
arg_parser.add_argument('--select_true_label', action='store_true')  # false: pl, True: true label
arg_parser.add_argument('--no_outer_selection', action='store_true',
                        help='whether outer loop has selection or not')
arg_parser.add_argument("--interval_val", type=int, default=10)

arg_parser.add_argument("--WARMSTART_EPOCH", type=int, default=100)
