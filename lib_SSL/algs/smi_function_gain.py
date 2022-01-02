# from MAML_SMI_version2.trust.trust.strategies.strategy import Strategy
import torch.nn.functional as F
from torch.utils.data import Subset
from maml.utils import DatasetSMI
from maml.utils import DatasetAugment
from trust.trust.strategies.smi import SMI
from trust.trust.strategies.strategy import Strategy
from trust.trust.utils.utils import *
import submodlib

def remove_overlap(groups, groups_gains, groups_pseudolabels, budget_class, excluded_set):
    '''
    :param groups:  the original bigger selected examples for each class, [[],[],[],[],[]]
    :param groups_gains:  [[],[],[],[],[]]
    :param groups_pseudolabels:  [[],[],[],[],[]]
    :param budget_class:  budget for each class
    :param excluded_set:
    :return:  su_all: extracted non-repeated examples: []
    '''
    num_class = len(groups)
    num_examples_per_class = len(groups[0])

    su_all = []   # stores extracted unique elements for all classes
    su_pl_all = []   # pseudolabels
    name = locals()

    for i in range(num_class):  # class 0 - 4 (for 5 classes)
        # print(G[i], "vs.")
        name[f"su_{i}"] = []   # su_1 = [], su_2 = [], su_3 = [], su_4 = []
        name[f"pl_{i}"] = []   # pl_1 = [], pl_2 = [], pl_3 = [], pl_4 = [],
        for j in range(num_examples_per_class):
            ele = groups[i][j]
            ele_gain = groups_gains[i][j]
            ele_pseudolabel = groups_pseudolabels[i][j]

            if ele in excluded_set:
                continue

            overlap_ele_gains = []   # stores the gains of all overlapping elements among other groups

            # find the index if e exists in other groups
            # if the current group is G[i], others are G[i-1], G[i-2], G[i-3], G[i-4] (5 classes, 5 groups totally)
            for shift in range(1, num_class):
                if ele in groups[i - shift]:
                    pos_ele_new = groups[i - shift].index(ele)
                    overlap_ele_gains.append(groups_gains[i-shift][pos_ele_new])
            # print(f"The index of overlapping elements:{overlap_ele_gains}")
            # if len(overlap_ele_gains)>0 and ele_gain == max(overlap_ele_gains):
            #     print("~~~~~~~ all gains for the same example:", ele_gain, "\t", overlap_ele_gains)
            # no overlapping elements in other groups or max gain
            if len(overlap_ele_gains) == 0 or ele_gain > max(overlap_ele_gains):
                name[f"su_{i}"].append(ele)
                name[f"pl_{i}"].append(ele_pseudolabel)

            if len(name[f"su_{i}"]) == budget_class:  # stop when su_0(su_1/su_2/su_3/su_4) arrives the budget
                break

        # print(name.get(f'su_{i}'))
        su_all.extend(name.get(f"su_{i}"))   # extend instead of append
        su_pl_all.extend(name.get(f"pl_{i}"))
    return su_all, su_pl_all


def fillBudget(select_idx_wo_overlap, select_pls_wo_overlap,
               select_idx, select_gains, select_pls,
               full_budget):
    num_class = len(select_idx)
    num_examples_per_class = len(select_idx[0])
    while(len(select_idx_wo_overlap) != full_budget):
        for i in range(num_class):
            for j in range(num_examples_per_class):
                if select_idx[i][j] not in select_idx_wo_overlap and select_gains[i][j] > 0:
                    select_idx_wo_overlap.append(select_idx[i][j])
                    select_pls_wo_overlap.append(select_pls[i][j])
                    if len(select_idx_wo_overlap) == full_budget:
                        return select_idx_wo_overlap, select_pls_wo_overlap
                    else:
                        break
    return select_idx_wo_overlap, select_pls_wo_overlap


def smi_function(support_inputs, support_targets,
                 query_inputs, query_targets,
                 unlabel_inputs, unlabeled_targets,
                 selection_option,
                 is_inner,
                 excluded_set,
                 model,
                 num_cls,
                 meta_train=True,
                 strategy_args={}):
    """
    :param support_inputs: SMI query set, it could be support set, for 3-way 5-shot, size is 15,3,32,32
    :param support_targets: label of SMI query set, Tensor:(15,)
    :param unlabel_inputs: unlabeled set, i.e., 300,3,32,32
    :param unlabeled_targets: true label of unlabeled set, i.e., Tensor:(300,)
    :param model: the backbone model
    :param num_cls: the number of classes
    :param strategy_args:
    :return:
            select_idx: indices of selected samples
            select_pls: their corresponding pseudo labels
    """

    train_set = DatasetSMI(support_inputs, support_targets)

    # data loader transform
    if meta_train:
        if selection_option == "same":
            if is_inner:
                smi_query_set = DatasetSMI(support_inputs, support_targets)
            else:
                smi_query_set = DatasetSMI(query_inputs, query_targets)

        elif selection_option == "cross":
            if is_inner:
                smi_query_set = DatasetSMI(query_inputs, query_targets)
            else:
                smi_query_set = DatasetSMI(support_inputs, support_targets)

        elif selection_option == "union":
            # concatenate the support set and query set: 45,3,32,32
            inputs_support_query = torch.cat([support_inputs, query_inputs], 0)
            targets_support_query = torch.cat([support_targets, query_targets], 0)
            smi_query_set = DatasetSMI(inputs_support_query, targets_support_query)
    else:
        # augment support set
        # train_set = DatasetAugment(support_inputs, support_targets)
        smi_query_set = DatasetSMI(support_inputs, support_targets)

    unlabeled_set = DatasetSMI(unlabel_inputs, unlabeled_targets)
    unlabeled_set = LabeledToUnlabeledDataset(unlabeled_set)

    smi_function = strategy_args['smi_function'] if 'smi_function' in strategy_args else "fl2mi"
    embedding_type = strategy_args['embedding_type'] if 'embedding_type' in strategy_args else "gradients"

    #compute embeddings of the unlabeled set out the class loop
    strategy_obj = Strategy(train_set, unlabeled_set, model, num_cls, strategy_args)
    if(embedding_type == "gradients"):
        unlabeled_data_embedding = strategy_obj.get_grad_embedding(unlabeled_set, True, "bias_linear")
    else: #use class scores
        unlabeled_data_embedding = strategy_obj.get_class_scores(unlabeled_set)

    budget_all = int(strategy_args['budget']) + len(excluded_set)
    budget_per_class = int(strategy_args['budget'] / num_cls)
    
    select_idx = []      # indice of selected examples in the unlabeled set
    select_pls = []      # pseudolabels of selected examples in the unlabeled set
    select_gains = []    # gains of selected examples in the unlabeled set
    
    # per class for loop
    for i in range(num_cls):
        # Find indices of the smi_query_set which have samples from class i
        val_class_idx = torch.where(smi_query_set.target == i)[0]
        val_class_subset = Subset(smi_query_set, val_class_idx)
        if(embedding_type == "gradients"):
            smi_query_data_embedding = strategy_obj.get_grad_embedding(val_class_subset, False, "bias_linear")
        else: #use class scores
            smi_query_data_embedding = torch.zeros(num_cls)
            smi_query_data_embedding[i] = 1
        query_sijs = submodlib.helper.create_kernel(X=smi_query_data_embedding.cpu().numpy(), X_rep=unlabeled_data_embedding.cpu().numpy(), metric="cosine", method="sklearn")
        if(smi_function == "fl2mi"):
            print("Using FL2MI for subset selection!")
            obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=smi_query_data_embedding.shape[0], 
                                                                      query_sijs=query_sijs, 
                                                                      queryDiversityEta=1)
        if(smi_function == "gcmi"):
            print("Using GCMI for subset selection!")
            obj = submodlib.GraphCutMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=smi_query_data_embedding.shape[0], 
                                                                      query_sijs=query_sijs,
                                                                      metric="cosine")

        greedyList = obj.maximize(budget=budget_all,optimizer="LazyGreedy", stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False)
        subset_idx_for_the_class = [x[0] for x in greedyList]
        selected_idx_class_gain = [x[1] for x in greedyList]                                                                    
        # smi selection for class i
        # strategy_sel = SMI(train_set, unlabeled_set, val_class_subset, model, num_cls, strategy_args) # todo: is budget used here?
        # subset_idx_for_the_class = strategy_sel.select(budget_all) # todo: occupy the GPU 0 for a fixed number
        # subset_idx_for_the_class, selected_idx_class_gain = strategy_sel.select(budget_all)
        select_idx.append(subset_idx_for_the_class)
        select_gains.append(selected_idx_class_gain)
        # a list, store smi pseudo labels as the current class of query_class_subset
        select_pls.append([i]*budget_all)

    # for i in range(5):
    #     print("\n****** GAIN: ", select_gains[i])

    select_idx_wo_overlap, select_pls_wo_overlap = remove_overlap(select_idx, select_gains, select_pls,
                                                                  budget_per_class, excluded_set)

    full_budget = strategy_args['budget']
    if len(select_idx_wo_overlap) != full_budget:
        return fillBudget(select_idx_wo_overlap, select_pls_wo_overlap,
                          select_idx, select_gains, select_pls,
                          full_budget)

    return select_idx_wo_overlap, select_pls_wo_overlap


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
    num_unlabeled = len(unlabeled_targets)
    num_select = len(selected_idx)
    num_select_wo_duplicate = gt_mask.sum(0)
    num_oods_select = 0
    if scenario == "distractor":
        labels_unlabeled_select = unlabeled_targets * gt_mask
        num_oods_select = (labels_unlabeled_select == -1).sum(dim=0)

    if scenario in ["woDistractor", "distractor"]:         # does not work for random selection for now
        # ====== log the selection statistics
        selected_targets = unlabeled_targets[selected_idx]     # todo: check the index whether matched or not
        num_selected_correct = torch.sum(selected_targets == selected_pseudolabels)
        select_stat = "{}, {}, {}, {}, {}".format(num_selected_correct, num_select_wo_duplicate, num_select, num_unlabeled, num_oods_select)
        print(f"+++++ Some statistics in the selection: {select_stat}") if verbose else None
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
    return loss_selected, select_stat


def smi_pl_comb(support_inputs, support_targets,
                query_inputs, query_targets,
                unlabel_inputs, unlabeled_targets,
                selection_option,
                is_inner,
                excluded_set,
                model_smi_copy,
                model, params,
                num_cls,
                meta_train,
                is_select_true_label,
                scenario,
                verbose,
                strategy_args={}):

    # 0.: first remove the unlabeled examples from the excluded_set (for outer loop selection)
    if excluded_set:
        unlabeled_size = len(unlabeled_targets)  # original size of unlabeled set
        rest_indices = list(set(list(range(unlabeled_size))) - excluded_set)
        unlabel_inputs = unlabel_inputs[rest_indices]
        unlabeled_targets = unlabeled_targets[rest_indices]

    # 1. get the index and pseudo labels of selected samples
    selected_idx, selected_pseudolabels = smi_function(support_inputs, support_targets,
                                                       query_inputs, query_targets,
                                                       unlabel_inputs, unlabeled_targets,
                                                       selection_option,
                                                       is_inner,
                                                       model_smi_copy,
                                                       num_cls,
                                                       meta_train=meta_train,
                                                       strategy_args=strategy_args)
    print("\nselected indices: ", selected_idx) if verbose else None

    selected_idx_tensor = torch.tensor(selected_idx).to(strategy_args['device'])  # Tensor:(30,)
    selected_pseudolabels = torch.tensor(selected_pseudolabels).to(strategy_args['device'])  # Tensor:(30,)

    # 2. calculate the loss of the selected unlabeled set
    loss_unlabel, select_stat = smi_pl_loss(unlabel_inputs, unlabeled_targets,
                                                    selected_idx_tensor, selected_pseudolabels,
                                                    model, params,
                                                    num_cls, is_select_true_label,
                                                    scenario,
                                                    verbose)
    return loss_unlabel, select_stat, selected_idx
