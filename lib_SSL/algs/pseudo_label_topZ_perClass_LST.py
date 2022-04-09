import torch
import torch.nn as nn
import torch.nn.functional as F

class PLtopZswn(nn.Module):
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
        # self.forward_swn = swn


    def sample_unlabel_feature(self, unlabeled_inputs, model, params):
        # self.model.eval() # tmp
        with torch.no_grad():
            outputs_unlabeled, features_unlabeled = model(unlabeled_inputs, params=params, keep_feat=True)
        y_probs = outputs_unlabeled.detach().softmax(1)  # predicted logits for unlabeled set
        pseudolabels = torch.argmax(y_probs, 1)

        return outputs_unlabeled, features_unlabeled, pseudolabels



    def sample_selection_hard(self, unlabeled_inputs, model, params, unlabeled_targets):
        # self.model.eval() # tmp
        with torch.no_grad():
            outputs_unlabeled = model(unlabeled_inputs, params=params)
        y_probs = outputs_unlabeled.detach().softmax(1)  # predicted logits for unlabeled set

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

        # ====== log the selection statistics
        selected_targets = unlabeled_targets[selected_idx]  # todo: check the index whether matched or not
        num_selected_correct = torch.sum(selected_targets == selected_pseudolabels)
        select_stat = "{}, {}, {}, {}, {}, {}".format(num_selected_correct, num_select_wo_duplicate, num_select,
                                                  num_unlabeled, num_oods_select, selected_scores)
        print(f"+++++ Some statistics in the selection: {select_stat}") if self.verbose else None
        # ======

        selected_unlabel_samples = unlabeled_inputs[selected_idx]
        # model.train() # tmp
        select_output, select_features = model(selected_unlabel_samples, params=params, keep_feat=True)

        return select_output, select_features, selected_pseudolabels, select_stat, selected_idx
               # 25,5               # 25,64,5,5

    def __make_one_hot(self, y, n_classes=3):  # todo: attention to the dimension
        return torch.eye(n_classes)[y].to(y.device)


    def computing_soft_weights(self, feat_s, label_s, feat_un, samples_num, swn_model):
        '''
        Computing soft-weights for unlabeled samples.
        '''
        # num_label_s = torch.argmax(label_s, axis=1)
        num_label_s = label_s
        w_list = []

        for i in range(self.num_cls):
            index_samples_i = torch.where(num_label_s==i)[0]
            feat_class_i = feat_s[index_samples_i]
            emb_class_i = torch.mean(feat_class_i, axis=0, keepdim=True)

            tile_emb_class_i = torch.tile(emb_class_i, (samples_num, 1, 1, 1))
            concat_emb = torch.cat([feat_un, tile_emb_class_i], axis=1) # channels

            w_i = torch.reshape(swn_model(concat_emb), (-1, 1))  # todo: check swn
            w_list.append(w_i)
        soft_weights = torch.cat(w_list, axis=1)

        return soft_weights


    def forward(self, x, x_label, unlabeled_inputs, model, params, unlabeled_targets,
                swn_model=None, hasSelection=True):

        if hasSelection:
            select_output, select_features, selected_pseudolabels, \
            select_stat, selected_idx = self.sample_selection_hard(unlabeled_inputs,
                                                                   model,
                                                                   params,
                                                                   unlabeled_targets)
            output_support, features_support = model(x, params=params, keep_feat=True)
            soft_weights = self.computing_soft_weights(features_support.detach(), x_label, select_features.detach(), self.topZ, swn_model)

            p_target = self.__make_one_hot(selected_pseudolabels, self.num_cls)
            select_output_new = nn.Softmax(dim=1)(soft_weights)*select_output
            losses_selected = -(p_target.detach() * F.log_softmax(select_output_new, 1)).sum(1)
            loss_selected = torch.sum(losses_selected) / len(selected_pseudolabels)

            print("+++++ PLtopZperClass the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None
            return loss_selected, select_stat, selected_idx.cpu().numpy(), selected_pseudolabels, soft_weights

        else:
            output_support, features_support = model(x, params=params, keep_feat=True)
            outputs_unlabeled, features_unlabeled, pseudolabels = self.sample_unlabel_feature(unlabeled_inputs, model, params)
            num_sample = outputs_unlabeled.shape[0]
            soft_weights = self.computing_soft_weights(features_support.detach(), x_label, features_unlabeled.detach(),
                                                       num_sample, swn_model)

            p_target = self.__make_one_hot(pseudolabels, self.num_cls)
            select_output_new = nn.Softmax(dim=1)(soft_weights) * outputs_unlabeled
            losses_selected = -(p_target.detach() * F.log_softmax(select_output_new, 1)).sum(1)
            loss_selected = torch.sum(losses_selected) / len(pseudolabels)

            print("+++++ PLtopZperClass the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None
            return loss_selected, pseudolabels, soft_weights


    def onlyUpdateBackbone(self, unlabeled_inputs, model, params,
                           selected_idx_updated, selected_pseudolabels, soft_weights):

        selected_unlabel_samples = unlabeled_inputs[selected_idx_updated]
        select_output = model(selected_unlabel_samples, params=params)

        p_target = self.__make_one_hot(selected_pseudolabels, self.num_cls)
        select_output_new = nn.Softmax(dim=1)(soft_weights)*select_output
        losses_selected = -(p_target.detach() * F.log_softmax(select_output_new, 1)).sum(1)
        loss_selected = torch.sum(losses_selected) / len(selected_pseudolabels)
        print("+++++ PLtopZperClass the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None

        return loss_selected

    def onlyUpdateBackbone_withOUTselection(self, unlabeled_inputs, model, params,
                           pseudolabels, soft_weights):
        output = model(unlabeled_inputs, params=params)

        p_target = self.__make_one_hot(pseudolabels, self.num_cls)
        select_output_new = nn.Softmax(dim=1)(soft_weights)*output
        losses_selected = -(p_target.detach() * F.log_softmax(select_output_new, 1)).sum(1)
        loss_selected = torch.sum(losses_selected) / len(pseudolabels)
        print("+++++ PLtopZperClass the SSL loss : {:.8f}.\n".format(loss_selected)) if self.verbose else None

        return loss_selected

