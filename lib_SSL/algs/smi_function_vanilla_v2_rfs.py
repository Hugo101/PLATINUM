import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from maml.utils import DatasetSMI, model_update, DatasetAugment
from trust.trust.strategies.smi import SMI
from trust.trust.strategies.strategy import Strategy
from trust.trust.utils.utils import *
import submodlib
import numpy as np


class SMIselection(nn.Module):
    def __init__(self, num_cls, select_true_label, scenario, selection_option, is_inner, verbose, freeze=True):
        super().__init__()
        self.num_cls = num_cls
        # self.budget = budget
        self.select_true_label = select_true_label  # only for debugging
        self.scenario = scenario
        self.verbose = verbose
        # self.batch_size = batch_size
        self.selection_option = selection_option
        self.is_inner = is_inner
        self.freeze = freeze

    def forward(self, support_inputs, support_targets,
                query_inputs, query_targets,
                unlabeled_inputs, unlabeled_targets,
                outputs_unlabeled,
                model, params,
                meta_train,
                excluded=[],
                strategy_args={},
                epoch=None):

        # # 0.: first remove the unlabeled examples from the excluded_set (for outer loop selection)
        if excluded:
            unlabeled_size = len(unlabeled_targets)  # original size of unlabeled set
            rest_indices = list(set(list(range(unlabeled_size))) - set(excluded))
            unlabeled_inputs = unlabeled_inputs[rest_indices]
            unlabeled_targets = unlabeled_targets[rest_indices]
            strategy_args['batch_size'] = len(rest_indices)

        # 1. get the index and pseudo labels of selected samples
        # model_smi_copy = copy.deepcopy(model)  # Deep copy the other exactly the same model
        # model_smi_copy = model_update(model_smi_copy, params)  # update model for this step # VITAL!!!
        selected_idx, selected_pseudolabels, selected_gains = self.smi_select(support_inputs, support_targets,
                                                                              query_inputs, query_targets,
                                                                              unlabeled_inputs, unlabeled_targets,
                                                                              outputs_unlabeled,
                                                                              None,
                                                                              model, params,
                                                                              meta_train=meta_train,
                                                                              strategy_args=strategy_args,
                                                                              epoch=epoch)
        print("\nselected indices: ", selected_idx) if self.verbose else None

        selected_idx_tensor = torch.tensor(selected_idx).to(strategy_args['device'])  # Tensor:(30,)
        selected_pseudolabels = torch.tensor(selected_pseudolabels).to(strategy_args['device'])  # Tensor:(30,)

        # 2. calculate the loss of the selected unlabeled set
        loss_unlabel, select_stat = self.smi_pl_loss(unlabeled_inputs, unlabeled_targets,
                                                     selected_idx_tensor, selected_pseudolabels, selected_gains,
                                                     model, params, self.freeze)

        return loss_unlabel, select_stat, selected_idx

    def __make_one_hot(self, y, n_classes=3):
        return torch.eye(n_classes)[y].to(y.device)

    def smi_select(self, support_inputs, support_targets,
                   query_inputs, query_targets,
                   unlabeled_inputs, unlabeled_targets,
                   unlabeled_data_embedding,
                   model_smi_copy,
                   model, params,
                   meta_train=True,
                   strategy_args={},
                   epoch=None):

        train_set = DatasetSMI(support_inputs, support_targets)
        # data loader transform
        if meta_train:
            if self.selection_option == "same":
                if self.is_inner:
                    smi_query_set = DatasetSMI(support_inputs, support_targets)
                else:
                    smi_query_set = DatasetSMI(query_inputs, query_targets)
            elif self.selection_option == "cross":
                if self.is_inner:
                    smi_query_set = DatasetSMI(query_inputs, query_targets)
                else:
                    smi_query_set = DatasetSMI(support_inputs, support_targets)
            elif self.selection_option == "union":
                # concatenate the support set and query set: 45,3,32,32
                inputs_support_query = torch.cat([support_inputs, query_inputs], 0)
                targets_support_query = torch.cat([support_targets, query_targets], 0)
                smi_query_set = DatasetSMI(inputs_support_query, targets_support_query)
        else:
            # augment support set
            # train_set = DatasetAugment(support_inputs, support_targets)
            smi_query_set = DatasetSMI(support_inputs, support_targets)

        unlabeled_set = DatasetSMI(unlabeled_inputs, unlabeled_targets)
        unlabeled_set = LabeledToUnlabeledDataset(unlabeled_set)

        smi_function = strategy_args['smi_function'] if 'smi_function' in strategy_args else "fl2mi"
        if epoch is None:
            embedding_type = strategy_args['embedding_type'] if 'embedding_type' in strategy_args else "gradients"
        else:
            embedding_type = "gradients" if epoch < 300 else "classScores"

        # compute embeddings of the unlabeled set out the class loop
        strategy_obj = Strategy(train_set, unlabeled_set, model, self.num_cls, strategy_args)
        if (embedding_type == "gradients"):
            unlabeled_data_embedding = strategy_obj.get_grad_embedding(params, unlabeled_set, True,
                                                                       "bias_linear")  # (250,8005)
        else:  # use class scores
            unlabeled_data_embedding = strategy_obj.get_class_scores(params, unlabeled_set)  # (250,5)

        # with torch.no_grad():
        #     unlabeled_data_embedding = model(unlabeled_inputs, params=params)
        # unlabeled_data_embedding = unlabeled_data_embedding.softmax(1)

        # budget_all = int(strategy_args['budget'])
        budget_per_class = int(strategy_args['budget'] / self.num_cls)

        select_idx = []
        select_pls = []
        select_gains = []
        # per class for loop
        for i in range(self.num_cls):
            # Find indices of the val_set which have samples from class i
            val_class_idx = torch.where(smi_query_set.target == i)[0]  # (20,)
            val_class_subset = Subset(smi_query_set, val_class_idx)

            if (embedding_type == "gradients"):
                smi_query_data_embedding = strategy_obj.get_grad_embedding(params, val_class_subset, False,
                                                                           "bias_linear")  # 20,8005
                query_sijs = submodlib.helper.create_kernel(X=smi_query_data_embedding.cpu().numpy(),
                                                            X_rep=unlabeled_data_embedding.cpu().numpy(),
                                                            metric="cosine",
                                                            method="sklearn")  # 250,20 for embedding_type gradients
                if (smi_function == "fl1mi" or smi_function == "logdetmi"):
                    data_sijs = submodlib.helper.create_kernel(X=unlabeled_data_embedding.cpu().numpy(),
                                                               metric="cosine", method="sklearn")
                if (smi_function == "logdetmi"):
                    query_query_sijs = submodlib.helper.create_kernel(X=smi_query_data_embedding.cpu().numpy(),
                                                                      metric="cosine", method="sklearn")

            else:  # use class scores
                smi_query_data_embedding = torch.zeros(1, self.num_cls)
                smi_query_data_embedding[0][i] = 1
                # query_sijs = submodlib.helper.create_kernel(X=smi_query_data_embedding.cpu().numpy(),
                #                                             X_rep=unlabeled_data_embedding.cpu().numpy(), metric="cosine",
                #                                             method="sklearn")   # 250,20 for embedding_type gradients, 250,1 for class score
                query_sijs = np.tensordot(smi_query_data_embedding.cpu().numpy(),
                                          unlabeled_data_embedding.cpu().numpy(), axes=([1], [1])).T

            if (smi_function == "fl2mi"):
                # print("Using FL2MI for subset selection!")
                obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                                 num_queries=
                                                                                 smi_query_data_embedding.shape[0],
                                                                                 query_sijs=query_sijs,
                                                                                 queryDiversityEta=1)
            if (smi_function == "gcmi"):
                # print("Using GCMI for subset selection!")
                obj = submodlib.GraphCutMutualInformationFunction(n=unlabeled_data_embedding.shape[0],  # 250
                                                                  num_queries=smi_query_data_embedding.shape[0],  # 1
                                                                  query_sijs=query_sijs,
                                                                  metric="cosine")

            if (smi_function == "fl1mi"):
                obj = submodlib.FacilityLocationMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                          num_queries=smi_query_data_embedding.shape[0],
                                                                          data_sijs=data_sijs,
                                                                          query_sijs=query_sijs,
                                                                          magnificationEta=1)

            if (smi_function == "logdetmi"):
                obj = submodlib.LogDeterminantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                        num_queries=smi_query_data_embedding.shape[0],
                                                                        data_sijs=data_sijs,
                                                                        query_sijs=query_sijs,
                                                                        query_query_sijs=query_query_sijs,
                                                                        magnificationEta=1,
                                                                        lambdaVal=1)

            greedyList = obj.maximize(budget=budget_per_class, optimizer="LazyGreedy", stopIfZeroGain=False,
                                      stopIfNegativeGain=False, verbose=False)
            subset_idx_for_the_class = [x[0] for x in greedyList]
            selected_idx_class_gain = [x[1] for x in greedyList]

            # smi selection for class i
            # strategy_sel = SMI(train_set, unlabeled_set, val_class_subset, model, num_cls, strategy_args)
            # subset_idx_for_the_class = strategy_sel.select(budget_all)
            # subset_idx_for_the_class, selected_idx_class_gain = strategy_sel.select(budget_all)
            select_idx.extend(subset_idx_for_the_class)
            select_gains.extend(selected_idx_class_gain)
            # a list, store smi pseudo labels as the current class of query_class_subset
            select_pls.extend([i] * budget_per_class)

        # select_idx, select_pls = remove_overlap(select_idx, select_pls, budget_per_class)
        select_gains_5 = [float("%.5f" % i) for i in select_gains]
        return select_idx, select_pls, select_gains_5

    # to add unlabel_outputs if to consider the the pseudo labels given from the model (in case)
    def smi_pl_loss(self, unlabeled_inputs, unlabeled_targets,
                    selected_idx, selected_pseudolabels, selected_gains,
                    model, params, freeze):

        gt_mask = torch.zeros_like(unlabeled_targets)
        gt_mask[selected_idx.long()] = 1  # make the selected index as 1
        # intermediate results
        # num_unlabeled = len(unlabeled_targets)
        num_unlabeled = len(unlabeled_targets)
        num_select = len(selected_idx)
        num_select_wo_duplicate = gt_mask.sum(0)
        num_oods_select = 0
        if self.scenario == "distractor":
            labels_unlabeled_select = unlabeled_targets * gt_mask
            num_oods_select = (labels_unlabeled_select == -1).sum(dim=0)

        if self.scenario in ["woDistractor", "distractor", "imbalance"]:  # does not work for random selection for now
            # ====== log the selection statistics
            selected_targets = unlabeled_targets[selected_idx]  # todo: check the index whether matched or not
            # print("************** selected target:", selected_targets)
            num_selected_correct = torch.sum(selected_targets == selected_pseudolabels)
            select_stat = "{}, {}, {}, {}, {}, {}, {}".format(num_selected_correct, num_select_wo_duplicate,
                                                              num_select, num_unlabeled, num_oods_select,
                                                              selected_targets.cpu().numpy(), selected_gains)
            print(f"+++++ Some statistics in the selection: {select_stat}") if self.verbose else None
            # ======

        selected_unlabel_samples = unlabeled_inputs[selected_idx]
        select_output = model(selected_unlabel_samples, params=params, freeze=freeze)
        if self.select_true_label:
            p_target = self.__make_one_hot(selected_targets, self.num_cls)
        else:
            p_target = self.__make_one_hot(selected_pseudolabels, self.num_cls)
        losses_selected = -(p_target.detach() * F.log_softmax(select_output, 1)).sum(1)
        loss_selected = torch.sum(losses_selected) / len(selected_idx)

        print("+++++ SMI the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None
        return loss_selected, select_stat

    def remove_overlap(self, groups, groups_pseudolabels, budget_class):
        '''
        :param groups: the original bigger selected examples for each class, [[],[],[],[],[]]
        :param groups_pseudolabels:  [[],[],[],[],[]]
        :param budget_class: budget for each class
        :return:
            su_all: extracted non-repeated examples: []
        '''
        num_class = len(groups)
        num_examples_per_class = len(groups[0])

        su_all = []  # stores extracted unique elements for all classes
        su_pl_all = []  # pseudo labels
        name = locals()

        for i in range(num_class):  # class 0 - 4 (for 5 classes)
            # print(G[i], "vs.")
            name[f"su_{i}"] = []  # su_1 = [], su_2 = [], su_3 = [], su_4 = []
            name[f"pl_{i}"] = []  # pl_1 = [], pl_2 = [], pl_3 = [], pl_4 = []
            for j in range(num_examples_per_class):
                ele = groups[i][j]
                ele_id = groups[i].index(ele)
                ele_pseudolabel = groups_pseudolabels[i][j]

                # if ele in excluded_set:
                #     continue

                overlap_ele_ids = []  # stores the indices of all overlapping elements among other groups

                # find the index if e exists in other groups
                # if the current group is G[i], others are G[i-1], G[i-2], G[i-3], G[i-4] (5 classes, 5 groups totally)
                for shift in range(1, num_class):
                    if ele in groups[i - shift]:
                        pos_ele_new = groups[i - shift].index(ele)
                        overlap_ele_ids.append(pos_ele_new)
                # print(f"The index of overlapping elements:{overlap_ele_ids}")
                # no overlapping elements in other groups or min rank
                if len(overlap_ele_ids) == 0 or ele_id < min(overlap_ele_ids):
                    name[f"su_{i}"].append(ele)
                    name[f"pl_{i}"].append(ele_pseudolabel)

                if len(name[f"su_{i}"]) == budget_class:  # stop when su_0(su_1/su_2/su_3/su_4) arrives the budget
                    break

            # print(name.get(f'su_{i}'))
            su_all.extend(name.get(f"su_{i}"))  # extend instead of append
            su_pl_all.extend(name.get(f"pl_{i}"))
        return su_all, su_pl_all
