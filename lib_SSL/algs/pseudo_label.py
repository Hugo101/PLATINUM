import torch
import torch.nn as nn
import torch.nn.functional as F


class PL(nn.Module):
    def __init__(self, num_cls, threshold, batch_size, select_true_label, scenario, verbose, support_unlabeled_avg_flag=False):
        super().__init__()
        self.num_cls = num_cls
        self.th = threshold
        self.select_true_label = select_true_label     # only for debugging
        self.scenario = scenario
        self.verbose = verbose
        self.support_unlabeled_avg_flag = support_unlabeled_avg_flag
        self.entropy = False
        self.batch_size = batch_size

    def forward(self, x, y_output, model, params, y_true, excluded=[]):
        y_probs = y_output.softmax(1)    # predicted logits for unlabeled set
        onehot_label_pred = self.__make_one_hot(y_probs.max(1)[1], self.num_cls).float()
        rest_mask = torch.ones_like(y_true)
        if excluded:
            rest_mask[list(excluded)] = 0     # only the rest unlabeled set after inner loop are 1
            onehot_label_pred = onehot_label_pred * rest_mask[:, None]
            y_probs = y_probs * rest_mask[:, None]

        selected_idx = (y_probs > self.th).nonzero()[:, 0].cpu().numpy()
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0]  # 0: all prediction < th, 1: > th, 1 denotes potential selected
        num_selected = gt_mask.sum(0)

        # ####### intermediate results
        unlabeled_prediction = y_probs
        if self.verbose:
            print(f"\n+++++ The num of samples with > current TH {self.th} is: {num_selected}")
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

        if self.scenario == "random":
            # ==== for OOD randomly selection
            num_oods = (y_true * rest_mask == -1).sum(dim=0)
            num_unlabeled = rest_mask.sum(0)
            num_oods_select = (y_true * rest_mask * gt_mask == -1).sum(dim=0)
            ratio_ood = "{}/{}={:.4f}".format(num_oods_select, num_selected,
                                              num_oods_select / num_selected)
            if self.verbose:
                print("=== OOD samples analysis:")
                print("====== The ratio of OOD samples in the unlabeled set is "
                      "{}/{}={:.4f}.".format(num_oods, num_unlabeled, num_oods / num_unlabeled))
                print(f"====== The ratio of OOD samples in the selected unlabeled set is {ratio_ood}.")
            # ====

        if self.scenario == "woDistractor":
            # ====== todo: does not work for random selection for now
            onehot_label_true = self.__make_one_hot(y_true, self.num_cls)
            onehot_label_pred_selected = gt_mask[:, None] * onehot_label_pred
            onehot_label_pred_selected_correct = onehot_label_true * onehot_label_pred_selected
            num_selected_correct = torch.sum(onehot_label_pred_selected_correct.sum(1))
            accu_among_selected = num_selected_correct / num_selected
            acc_slct = "{}/{}={:.4f}".format(num_selected_correct, num_selected, accu_among_selected)
            print(f"+++++ The accu in the selected samples: {acc_slct}") if self.verbose else None
            # ======

        if self.entropy:
            lt_mask = 1 - gt_mask    # logical not
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

        # model.update_batch_stats(False)  # todo: unclear!!!!!!!!!!!!!!! might not be used
        # output = model(x)
        output = model(x, params=params)
        # loss = (-(p_target.detach() * F.log_softmax(y_output, 1)).sum(1)*mask).mean()
        losses_mask = -(p_target.detach() * F.log_softmax(output, 1)).sum(1)
        # model.update_batch_stats(True)

        if self.entropy:
            loss = torch.sum(losses_mask) / len(y_true)
        else:
            if torch.sum(losses_mask) == 0:
                loss = 0
            else:
                loss = torch.sum(losses_mask) / num_selected

        # =====
        # unlabeled_selected_loss = torch.sum(losses_mask * gt_mask)
        # unlabeled_unselected_loss = torch.sum(losses_mask) - unlabeled_selected_loss
        # print("++++++ the loss of selected samples vs. the loss of unselected samples: "
        #       "{:.8f} vs. {:.8f}.".format(unlabeled_selected_loss, unlabeled_unselected_loss))
        # =====

        print("++++++ the loss of selected unlabeled samples: {:.8f}.\n".format(loss)) if self.verbose else None
        if self.scenario == "woDistractor":
            return loss, 0, acc_slct, selected_idx
        if self.scenario == "random":
            return loss, ratio_ood, None, selected_idx

    def __make_one_hot(self, y, n_classes=3):    # todo: attention to the dimension
        return torch.eye(n_classes)[y].to(y.device)
