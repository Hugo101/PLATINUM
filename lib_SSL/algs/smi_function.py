import torch.nn.functional as F
from torch.utils.data import Subset
from maml.utils import DatasetSMI
from maml.utils import DatasetAugment
from trust.trust.strategies.smi import SMI
from trust.trust.utils.utils import *


def remove_overlap(groups, budget_class, excluded_set):
    '''
    :param groups: the original bigger selected examples for each class, [[],[],[],[],[]]
    :param budget_class: budget for each class
    :return:
        su_all: extracted non-repeated examples: []
    '''
    su_all = []  # stores extracted unique elements for all classes
    name = locals()
    num_class = len(groups)
    for i in range(num_class):  # class 0 - 4 (for 5 classes)
        # print(G[i], "vs.")
        name[f"su_{i}"] = []   # su_1 = [], su_2 = [], su_3 = [], su_4 = []
        for ele in groups[i]:
            if ele in excluded_set:
                continue

            overlap_ele_ids = []   # stores the indices of all overlapping elements among other groups

            # find the index if e exists in other groups
            # if the current group is G[i], others are G[i-1], G[i-2], G[i-3], G[i-4] (5 classes, 5 groups totally)
            for shift in range(1, num_class):
                if ele in groups[i - shift]:
                    overlap_ele_ids.append(groups[i - shift].index(ele))
            # print(f"The index of overlapping elements:{overlap_ele_ids}")
            if len(overlap_ele_ids) == 0:  # no overlapping elements in other groups
                name[f"su_{i}"].append(ele)
            else:
                name[f"su_{i}"].append(ele) if groups[i].index(ele) < min(overlap_ele_ids) else None

            if len(name[f"su_{i}"]) == budget_class:  # stop when su_0(su_1/su_2/su_3/su_4) arrives the budget
                break

        # print(name.get(f'su_{i}'))
        su_all.extend(name.get(f"su_{i}"))   # extend instead of append
    return su_all


def smi_function(smi_query_inputs, smi_query_targets,
                 unlabel_inputs, unlabeled_targets,
                 excluded_set,
                 model,
                 num_cls,
                 meta_train=True,
                 strategy_args={}):
    """
    :param smi_query_inputs: SMI query set, it could be support set, for 3-way 5-shot, size is 15,3,32,32
    :param smi_query_targets: label of SMI query set, Tensor:(15,)
    :param unlabel_inputs: unlabeled set, i.e., 300,3,32,32
    :param unlabeled_targets: true label of unlabeled set, i.e., Tensor:(300,)
    :param model: the backbone model
    :param num_cls: the number of classes
    :param strategy_args:
    :return:
            select_subset_idx: indices of selected samples
            select_subset_pseudolabels: their corresponding pseudo labels
    """
    # data loader transform
    if meta_train:
        train_set = DatasetSMI(smi_query_inputs, smi_query_targets)
        # inputs_support_query = torch.cat([inputs, test_inputs], 0)  # concatenate the support set and query set: 45,3,32,32
        # targets_support_qeury = torch.cat([targets, test_targets], 0)
        # train_set = DatasetSMI(inputs_support_query, targets_support_qeury)
    else:
        # augment support set
        # train_set = DatasetAugment(smi_query_inputs, smi_query_targets)
        train_set = DatasetSMI(smi_query_inputs, smi_query_targets)

    unlabeled_set = DatasetSMI(unlabel_inputs, unlabeled_targets)
    unlabeled_set = LabeledToUnlabeledDataset(unlabeled_set)
    val_set = train_set
    budget_all = int(strategy_args['budget']) + len(excluded_set)
    budget_per_class = int(strategy_args['budget'] / num_cls)

    select_subset_idx = []
    select_subset_pseudolabels = []
    select_subset_gains = []
    # per class for loop
    for i in range(num_cls):
        # Find indices of the val_set which have samples from class i
        val_class_idx = torch.where(val_set.target == i)[0]
        val_class_subset = Subset(val_set, val_class_idx)
        # smi selection for class i
        strategy_sel = SMI(train_set, unlabeled_set, val_class_subset, model, num_cls, strategy_args)
        # subset_idx_for_the_class = strategy_sel.select(budget_all) # todo: occupy the GPU 0 for a fixed number
        subset_idx_for_the_class, selected_idx_class_gain = strategy_sel.select(budget_all)
        select_subset_idx.append(subset_idx_for_the_class)
        select_subset_gains.append(selected_idx_class_gain)
        # select_subset_idx.extend(subset_idx_for_the_class)
        # a list, store smi pseudo labels as the current class of query_class_subset
        select_subset_pseudolabels.extend([i]*budget_per_class)

    select_subset_idx_wo_overlap = remove_overlap(select_subset_idx, budget_per_class, excluded_set)

    # selected_unlabeledSet = Subset(unlabeled_set, unlabeledSet_subset_idx)
    return select_subset_idx_wo_overlap, select_subset_pseudolabels


def __make_one_hot(y, n_classes=3):  # todo: attention to the dimension
    return torch.eye(n_classes)[y].to(y.device)


# to add unlabel_outputs if to consider the the pseudo labels given from the model (in case)
def smi_pl_loss(unlabeled_inputs, unlabeled_targets,
                selected_idx, selected_pseudolabels,
                model, params,
                num_cls, is_select_true_label,
                scenario,
                verbose):
    gt_mask = torch.zeros_like(unlabeled_targets)
    gt_mask[selected_idx.long()] = 1  # make the selected index as 1

    # intermediate results
    num_unlabeled_select = gt_mask.sum(0)
    if scenario == "random":
        # ==== for OOD randomly selection
        num_oods = (unlabeled_targets == -1).sum(dim=0)
        num_unlabeled = len(unlabeled_targets)
        labels_unlabeled_select = unlabeled_targets * gt_mask
        num_oods_select = (labels_unlabeled_select == -1).sum(dim=0)
        ratio_ood = "{}/{}={:.4f}".format(num_oods_select, num_unlabeled_select, num_oods_select / num_unlabeled_select)
        if verbose:
            print("\n=== OOD samples analysis:")
            print("====== The ratio of OOD samples in the unlabeled set is: "
                  "{}/{}={:.4f}.".format(num_oods, num_unlabeled, num_oods / num_unlabeled))
            print(f'====== The ratio of OOD samples in the selected unlabeled set is: {ratio_ood}')
        # ====

    if scenario == "woDistractor":
        # ====== log the accuracy of selected samples
        # todo: does not work for random selection for now
        selected_targets = unlabeled_targets[selected_idx]
        num_selected_correct = torch.sum(selected_targets == selected_pseudolabels)
        accu_among_selected = num_selected_correct / num_unlabeled_select
        acc_slct = "{}/{}={:.4f}".format(num_selected_correct, num_unlabeled_select, accu_among_selected)
        if verbose:
            print(f"+++++ The accu in the selected samples: {acc_slct}")
        # ======

    selected_unlabel_samples = unlabeled_inputs[selected_idx]
    select_output = model(selected_unlabel_samples, params=params)
    if is_select_true_label:
        p_target = __make_one_hot(selected_targets, num_cls)
    else:
        p_target = __make_one_hot(selected_pseudolabels, num_cls)
    losses_selected = -(p_target.detach() * F.log_softmax(select_output, 1)).sum(1)
    loss_selected = torch.sum(losses_selected) / len(selected_idx)

    print("+++++ the SSL loss : {:.8f}.\n".format(loss_selected)) if verbose else None
    if scenario == "woDistractor":
        return loss_selected, 0, acc_slct
    if scenario == "random":
        return loss_selected, ratio_ood, None


def smi_pl_comb(smi_query_inputs, smi_query_targets,
                unlabel_inputs, unlabeled_targets,
                excluded_set,
                model_smi_copy,
                model, params,
                num_cls,
                meta_train,
                is_select_true_label,
                scenario,
                verbose,
                strategy_args={}):
    # 1. get the index and pseudo labels of selected samples
    selected_idx, selected_pseudolabels = smi_function(smi_query_inputs, smi_query_targets,
                                                       unlabel_inputs, unlabeled_targets,
                                                       excluded_set,
                                                       model_smi_copy,
                                                       num_cls,
                                                       meta_train=meta_train,
                                                       strategy_args=strategy_args)
    print("\nselected indices: ", selected_idx) if verbose else None

    selected_idx = torch.tensor(selected_idx).to(strategy_args['device'])  # Tensor:(30,)
    selected_pseudolabels = torch.tensor(selected_pseudolabels).to(strategy_args['device'])  # Tensor:(30,)

    # 2. calculate the loss of the selected unlabeled set
    loss_unlabel, ratio_ood, acc_slct = smi_pl_loss(unlabel_inputs, unlabeled_targets,
                                                    selected_idx, selected_pseudolabels,
                                                    model, params,
                                                    num_cls, is_select_true_label,
                                                    scenario,
                                                    verbose)
    return loss_unlabel, ratio_ood, acc_slct, selected_idx
