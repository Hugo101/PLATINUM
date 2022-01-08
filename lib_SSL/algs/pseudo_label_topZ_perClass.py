import torch
import torch.nn as nn
import torch.nn.functional as F


class PLtopZ(nn.Module):
    def __init__(self, num_cls, num_topZ, batch_size, select_true_label, scenario, verbose, threshold=None):
        super().__init__()
        self.num_cls = num_cls
        self.topZ = num_topZ
        self.select_true_label = select_true_label  # only for debugging
        self.scenario = scenario
        self.verbose = verbose
        self.threshold = threshold
        self.entropy = False
        self.batch_size = batch_size

    def forward(self, unlabeled_inputs, model, params, unlabeled_targets, excluded=[]):
        # 0.: first remove the unlabeled examples from the excluded_set (for outer loop selection)
        if excluded:
            unlabeled_size = len(unlabeled_targets)  # original size of unlabeled set
            rest_indices = list(set(list(range(unlabeled_size))) - set(excluded))
            unlabeled_inputs = unlabeled_inputs[rest_indices]
            unlabeled_targets = unlabeled_targets[rest_indices]

        # self.model.eval() # tmp
        with torch.no_grad():
            outputs_unlabeled = model(unlabeled_inputs, params=params)
        y_probs = outputs_unlabeled.detach().softmax(1)  # predicted logits for unlabeled set

        # if excluded:
        #     rest_mask = torch.ones_like(unlabeled_targets)
        #     rest_mask[list(excluded)] = 0
        #     y_probs = y_probs * rest_mask[:, None]

        # step 1: top Z selection (per class topZ)
        budget_per_class = self.topZ // self.num_cls
        selected_scores, selected_idx_tensor = torch.topk(y_probs, budget_per_class, dim=0, largest=True, sorted=True)
        selected_idx = selected_idx_tensor.t().reshape(-1).detach()
        selected_pls = []
        for i in range(self.num_cls):
            selected_pls.extend([i] * budget_per_class)
        selected_pseudolabels = torch.tensor(selected_pls).to(y_probs.device)  # pseudo labels

        gt_mask = torch.zeros_like(unlabeled_targets)  # used to store the indices of final selected examples
        gt_mask[selected_idx] = 1   # the positions of selected samples are assigned to be 1, 0 otherwise

        num_unlabeled = len(unlabeled_targets)
        num_select = len(selected_idx)
        num_select_wo_duplicate = gt_mask.sum(0)
        num_oods_select = 0
        if self.scenario == "distractor":
            labels_unlabeled_select = unlabeled_targets * gt_mask
            num_oods_select = (labels_unlabeled_select == -1).sum(dim=0)

        if self.scenario in ["woDistractor", "distractor"]:
            # ====== log the selection statistics
            selected_targets = unlabeled_targets[selected_idx]  # todo: check the index whether matched or not
            num_selected_correct = torch.sum(selected_targets == selected_pseudolabels)
            select_stat = "{}, {}, {}, {}, {}, {}".format(num_selected_correct, num_select_wo_duplicate, num_select,
                                                      num_unlabeled, num_oods_select, selected_scores)
            print(f"+++++ Some statistics in the selection: {select_stat}") if self.verbose else None
            # ======

        selected_unlabel_samples = unlabeled_inputs[selected_idx]
        # model.train() # tmp
        select_output = model(selected_unlabel_samples, params=params)
        if self.select_true_label:
            p_target = self.__make_one_hot(selected_targets, self.num_cls)
        else:
            p_target = self.__make_one_hot(selected_pseudolabels, self.num_cls)
        losses_selected = -(p_target.detach() * F.log_softmax(select_output, 1)).sum(1)
        loss_selected = torch.sum(losses_selected) / len(selected_idx)

        print("+++++ PLtopZperClass the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None
        return loss_selected, select_stat, selected_idx.cpu().numpy()


    def __make_one_hot(self, y, n_classes=3):  # todo: attention to the dimension
        return torch.eye(n_classes)[y].to(y.device)
