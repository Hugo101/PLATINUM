import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

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
        onehot_label_pred = self.__make_one_hot(y_probs.max(1)[1], self.num_cls).float()  # predicated one-hot label
        gt_mask = torch.zeros_like(unlabeled_targets)   # used to store the indices of final selected examples

        # step 1: top Z selection
        selected_idx = torch.topk(y_probs.max(1)[0], self.topZ)[1]
        gt_mask[selected_idx] = 1   # the positions of selected samples are assigned to be 1, 0 otherwise

        # (optional) step 2: threshold selection further
        if self.threshold:
            y_probs = y_probs * gt_mask[:, None]
            selected_idx = (y_probs > self.threshold).nonzero()[:, 0]
            gt_mask = (y_probs > self.threshold).float()
            gt_mask = gt_mask.max(1)[0]  # 0: all prediction < th, 1: > th, 1 denotes potential selected

        num_unlabeled = len(unlabeled_targets)
        num_select = len(selected_idx)
        num_select_wo_duplicate = gt_mask.sum(0)

        # (optional) intermediate results
        unlabeled_prediction = y_probs
        if self.verbose:
            print(f"+++++ The highest confidence is {torch.max(unlabeled_prediction).detach().cpu().item()}")
            num_geq_05 = torch.sum(unlabeled_prediction.max(1)[0] > 0.5).item()
            print(f"+++++ The num of samples with high confidence > 0.5  is: {num_geq_05}")
            num_geq_06 = torch.sum(unlabeled_prediction.max(1)[0] > 0.6).item()
            print(f"+++++ The num of samples with high confidence > 0.6  is: {num_geq_06}")
            num_geq_07 = torch.sum(unlabeled_prediction.max(1)[0] > 0.7).item()
            print(f"+++++ The num of samples with high confidence > 0.7  is: {num_geq_07}")
            num_geq_08 = torch.sum(unlabeled_prediction.max(1)[0] > 0.8).item()
            print(f"+++++ The num of samples with high confidence > 0.8  is: {num_geq_08}")
            num_geq_09 = torch.sum(unlabeled_prediction.max(1)[0] > 0.9).item()
            print(f"+++++ The num of samples with high confidence > 0.9  is: {num_geq_09}")
            num_geq_095 = torch.sum(unlabeled_prediction.max(1)[0] > 0.95).item()
            print(f"+++++ The num of samples with high confidence > 0.95 is: {num_geq_095}")

        num_oods_select = 0  # this number will update if for "distractor"
        if self.scenario == "distractor":
            num_oods_select = (unlabeled_targets * gt_mask == -1).sum(dim=0)

        if self.scenario in ["woDistractor", "distractor"]:
            # ====== log the selection statistics
            onehot_label_true = self.__make_one_hot(unlabeled_targets, self.num_cls)         # 350,5 (350 #unlabel)
            onehot_label_pred_selected = gt_mask[:, None] * onehot_label_pred                    # 350,5
            onehot_label_pred_selected_correct = onehot_label_true * onehot_label_pred_selected  # 350,5
            num_selected_correct = torch.sum(onehot_label_pred_selected_correct.sum(1))

            pure_pred_select_onehot = onehot_label_pred_selected[onehot_label_pred_selected.sum(dim=1) != 0]
            pure_pls = pure_pred_select_onehot.max(1)[1].detach().cpu().numpy()
            class_counts = Counter(pure_pls)

            select_stat = "{}, {}, {}, {}, {}, {}".format(num_selected_correct, num_select_wo_duplicate, num_select,
                                                      num_unlabeled, num_oods_select, class_counts)
            print(f"+++++ Some statistics in the selection: {select_stat}") if self.verbose else None
            # ======

        if self.entropy:
            lt_mask = 1 - gt_mask  # logical not
            # if lt_mask is always 0, then this is cross entropy completely and exactly
            # todo: does not work for random selection for now
            if self.select_true_label:
                p_target = gt_mask[:, None] * 10 * onehot_label_true + lt_mask[:, None] * y_probs
            else:
                p_target = gt_mask[:, None] * 10 * onehot_label_pred + lt_mask[:, None] * y_probs
        else:
            if self.select_true_label:
                p_target = gt_mask[:, None] * onehot_label_true
            else:
                p_target = gt_mask[:, None] * onehot_label_pred

        torch.cuda.empty_cache()
        # selected_unlabel_samples = unlabeled_inputs[selected_idx]
        # model.train() # tmp
        output = model(unlabeled_inputs, params=params)  # only one batch
        losses_selected = -(p_target.detach() * F.log_softmax(output, 1)).sum(1)

        if self.entropy:
            loss_selected = torch.sum(losses_selected) / len(unlabeled_targets)
        else:
            if torch.sum(losses_selected) == 0:
                loss_selected = 0
            else:
                loss_selected = torch.sum(losses_selected) / num_select

        print("+++++ PLtopZ the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None
        return loss_selected, select_stat, selected_idx.cpu().numpy()


    def __make_one_hot(self, y, n_classes=3):  # todo: attention to the dimension
        return torch.eye(n_classes)[y].to(y.device)
