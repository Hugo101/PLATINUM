import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
import time
from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from maml.utils import tensors_to_device, compute_accuracy
from lib_SSL.config import config
import pickle
from configuration import arg_parser

args = arg_parser.parse_args()

__all__ = ['ModelAgnosticMetaLearningLST']

# other baselines
if args.ssl_algo == 'LST':
    from lib_SSL.algs.pseudo_label_topZ_perClass_LST import PLtopZswn
    ssl_obj = PLtopZswn(args.num_ways, args.pl_num_topz, args.pl_batch_size, args.select_true_label,
                        args.scenario, args.verbose, args.pl_threshold)

WARMSTART_EPOCH = args.WARMSTART_EPOCH
WARMSTART_ITER = 10000
WARM = 0
WARM_inner = 1
WARM_inner_eval = 1

class ModelAgnosticMetaLearningLST(object):
    def __init__(self, model, swn_model, optimizer=None, optimizer_swn=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1,
                 num_adaptation_steps_test=10,
                 scheduler=None,
                 loss_function=F.cross_entropy,
                 device=None):
        self.model = model.to(device=device)
        self.swn_model = swn_model.to(device=device)
        self.optimizer = optimizer
        self.optimizer_swn = optimizer_swn

        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.num_adaptation_steps_test = num_adaptation_steps_test
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.support_unlabeled_avg = False  # added

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                                                             dtype=param.dtype, device=self.device,
                                                             requires_grad=learn_step_size)) for (name, param)
                                         in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                                          device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group(
                {'params': self.step_size.values() if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                                         for group in self.optimizer.param_groups])


    def train(self, dataloader, max_batches=500, batch_size=1, verbose=True, progress=0, **kwargs):
        result_per_epoch = {
                'num_tasks': [],
                'inner_losses_labeled': [],
                'inner_losses_unlabeled': [],
                'inner_losses': [],
                'outer_losses': [],
                'outer_losses_unlabeled': [],
                'mean_outer_loss': [],
                # store inner and outer stat_selected
                'stat_selected': [],
                "accuracy_change_per_task": [],
                "accuracies_after": [],
        }
        with tqdm(total=max_batches*batch_size, disable=False, **kwargs) as pbar:
            batch_id = 1
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(batch_size)

                postfix = {'outer loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy (query)'] = '{0:.4f}'.format(np.mean(results['accuracies_after']))

                pbar.set_postfix(**postfix)

                if verbose:
                    print(f"\niteration: {progress*batch_id}")
                    print("inner loss (labeled): \n{}".format(results['inner_losses_labeled']))
                    print("inner loss (unlabeled): \n{}".format(results['inner_losses_unlabeled']))
                    print("inner loss (labeled+unlabeled): \n{}".format(results['inner_losses']))
                    print("inner and outer stat_selected:\n{}".format(results['stat_selected']))


                for k,v in results.items():
                    result_per_epoch[k].append(v)

                batch_id += 1

        return result_per_epoch


    def train_iter(self, dataloader, max_batches=500):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                               'optimizer is `None`. In order to train `{0}`, you must '
                               'specify a Pytorch optimizer as the argument of `{0}` '
                               '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                               'parameters(), lr=0.01), ...).'.format(__class__.__name__))

        num_batches = 0
        self.model.train()  # key point
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break
                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()
                self.optimizer_swn.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                # torch.cuda.empty_cache()
                outer_loss, results, outer_loss_r = self.get_outer_loss(batch, self.num_adaptation_steps)
                yield results

                with torch.autograd.set_detect_anomaly(True):
                    outer_loss.backward(retain_graph=True)
                    # outer_loss_r.backward(inputs=list(self.swn_model.parameters()))  # important !!!
                    outer_loss_r.backward()

                    self.optimizer.step()
                    self.optimizer_swn.step()

                num_batches += 1


    def get_outer_loss(self, batch, num_adapt_steps):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, query_targets = batch['test']
        num_tasks = query_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses_labeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses_unlabeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),

            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'outer_losses_unlabeled': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,

            # store inner and outer stat_selected
            'stat_selected': np.empty((num_adapt_steps+1, num_tasks), dtype=object), # include outer loop
            "accuracy_change_per_task": np.zeros((2, num_tasks), dtype=np.float32),
            # accu before inner loop
            # accu of support after inner loop
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32),  # accu of query set after outer loop
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        mean_outer_loss_retrain = torch.tensor(0., device=self.device)
        for task_id, (support_inputs, support_targets, query_inputs, query_targets, unlabeled_inputs, unlabeled_targets) \
                in enumerate(zip(*batch['train'], *batch['test'], *batch['unlabeled'])):
            # print(f"task_id:{task_id}, unlabeled targets: {unlabeled_targets}")
            # inner loop
            params, adaptation_results, params_retrain = self.adapt(support_inputs, support_targets,
                                                    query_inputs, query_targets,
                                                    unlabeled_inputs, unlabeled_targets,
                                                    is_classification_task=True,
                                                    num_adaptation_steps=num_adapt_steps,
                                                    step_size=self.step_size,
                                                    first_order=self.first_order,
                                                    meta_train=True,
                                                    warm_step=args.WARM_inner,
                                                    re_train_step=args.re_train_step)

            results['inner_losses_labeled'][:, task_id] = adaptation_results['inner_losses_labeled']
            results['inner_losses_unlabeled'][:, task_id] = adaptation_results['inner_losses_unlabeled']
            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results["accuracy_change_per_task"][0, task_id] = adaptation_results['accuracy_before']
            results["accuracy_change_per_task"][1, task_id] = adaptation_results['accuracy_support']

            # ############### outer loop
            with torch.set_grad_enabled(self.model.training):
                query_logits = self.model(query_inputs, params=params)
                outer_loss = self.loss_function(query_logits, query_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss
                results["accuracies_after"][task_id] = compute_accuracy(query_logits, query_targets)

                query_logits_r = self.model(query_inputs, params=params_retrain)
                outer_loss_r = self.loss_function(query_logits_r, query_targets)
                mean_outer_loss_retrain += outer_loss_r
                # results["accuracies_after"][task_id] = compute_accuracy(query_logits_r, query_targets)

        mean_outer_loss = mean_outer_loss.div(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        mean_outer_loss_retrain = mean_outer_loss_retrain.div(num_tasks)

        return mean_outer_loss, results, mean_outer_loss_retrain


    def adapt(self, support_inputs, support_targets,
              query_inputs, query_targets,
              unlabeled_inputs, unlabeled_targets,
              is_classification_task=True,
              num_adaptation_steps=1,
              step_size=0.1,
              first_order=False,
              meta_train=True,
              warm_step=1,
              re_train_step=2,
              ):

        params = None
        results = {'inner_losses_labeled': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'inner_losses_unlabeled': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'inner_losses': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'stat_selected': [],
                   }

        selected_ids_inner = []

        for step in range(num_adaptation_steps):
            print(f"\n++++++ At Inner Step {step+1}:") if args.verbose else None
            outputs_support = self.model(support_inputs, params=params)
            # # Supervised Loss
            loss_support = self.loss_function(outputs_support, support_targets, reduction="mean")

            if step < warm_step or step>re_train_step:   # mimic the pretrained model, (+ fine tunning steps)
                loss_unlabeled, select_stat = 0, 0

            elif step == warm_step:   # hard selection and soft weight net
                if meta_train: # without selection
                    # print("without selection ##########################")
                    loss_unlabeled, pseudolabels, soft_weights = ssl_obj(support_inputs, support_targets,
                                                                         unlabeled_inputs,
                                                                         self.model, params,
                                                                         unlabeled_targets,
                                                                         self.swn_model,
                                                                         hasSelection=False)

                else:
                    loss_unlabeled, select_stat, selected_ids, \
                    selected_pseudolabels, soft_weights = ssl_obj(support_inputs, support_targets,
                                                                  unlabeled_inputs,
                                                                  self.model, params,
                                                                  unlabeled_targets,
                                                                  self.swn_model)

            else:      # step in [warm_step+1, re_train_step]: #  re_train_step
                if meta_train: # without selection
                    # print("without selection %%%%%%%%%%%%%%")
                    loss_unlabeled = ssl_obj.onlyUpdateBackbone_withOUTselection(unlabeled_inputs,
                                                                                 self.model, params,
                                                                                 pseudolabels, soft_weights)
                else:
                    loss_unlabeled = ssl_obj.onlyUpdateBackbone(unlabeled_inputs,
                                                                self.model, params,
                                                                selected_ids, selected_pseudolabels, soft_weights)

            inner_loss = loss_support + loss_unlabeled

            results['inner_losses_labeled'][step] = loss_support.item()
            results['inner_losses_unlabeled'][step] = loss_unlabeled
            results['inner_losses'][step] = inner_loss

            if (step == 0) and is_classification_task:
                # acc before inner loop training 10-14-2021
                results['accuracy_before'] = compute_accuracy(outputs_support, support_targets)

            if (step == num_adaptation_steps-1) and is_classification_task:
                # acc for support set after inner loop training
                results['accuracy_support'] = compute_accuracy(outputs_support, support_targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                                                step_size=step_size, params=params,
                                                first_order=(not self.model.training) or first_order)

            if step <= re_train_step:
                params_retrain = params

        return params, results, params_retrain


    def get_outer_loss_evaluate(self, batch, num_adapt_steps, progress=0):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, query_targets = batch['test']
        num_tasks = query_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses_labeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses_unlabeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            # 'outer_losses_unlabeled': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'stat_selected': np.empty((num_adapt_steps, num_tasks), dtype=object),
            "accuracy_change_per_task": np.zeros((2, num_tasks), dtype=np.float32),
            # accu before inner loop
            # accu of support after inner loop
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32),     # accu of query set after outer loop
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        mean_outer_loss_retrain = torch.tensor(0., device=self.device)
        for task_id, (support_inputs, support_targets, query_inputs, query_targets, unlabeled_inputs, unlabeled_targets) \
                in enumerate(zip(*batch['train'], *batch['test'], *batch['unlabeled'])):
            params, adaptation_results, params_retrain = self.adapt(support_inputs, support_targets,
                                                                       None, None,
                                                                       unlabeled_inputs, unlabeled_targets,
                                                                       is_classification_task=True,
                                                                       num_adaptation_steps=num_adapt_steps,
                                                                       step_size=self.step_size,
                                                                       first_order=self.first_order,
                                                                       meta_train=False,
                                                                       warm_step=args.WARM_inner_test,
                                                                       re_train_step=args.re_train_step_test)

            results['inner_losses_labeled'][:, task_id] = adaptation_results['inner_losses_labeled']
            results['inner_losses_unlabeled'][:, task_id] = adaptation_results['inner_losses_unlabeled']
            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results['accuracy_change_per_task'][0, task_id]  = adaptation_results['accuracy_before']
            results['accuracy_change_per_task'][1, task_id] = adaptation_results['accuracy_support']

            with torch.set_grad_enabled(self.model.training):    # For query set. meta-train: True, meta-val/test:False
                query_logits = self.model(query_inputs, params=params)
                outer_loss = self.loss_function(query_logits, query_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

                query_logits_r = self.model(query_inputs, params=params_retrain)
                outer_loss_r = self.loss_function(query_logits_r, query_targets)
                mean_outer_loss_retrain += outer_loss_r

            results['accuracies_after'][task_id] = compute_accuracy(query_logits, query_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        mean_outer_loss_retrain.div_(num_tasks)

        return mean_outer_loss, results, mean_outer_loss_retrain


    def evaluate(self, dataloader, max_batches=500, batch_size=4, verbose=True, progress=0, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        result_per_epoch = {
            'num_tasks': [],
            'inner_losses_labeled': [],
            'inner_losses_unlabeled': [],
            'inner_losses': [],
            'outer_losses': [],
            'mean_outer_loss': [],
            # store inner and outer stat_selected
            'stat_selected': [],
            "accuracy_change_per_task": [],
            "accuracies_after": [],
        }
        # accuracies
        accuracies = []
        with tqdm(total=max_batches*batch_size, disable=False, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches, progress=progress):
                pbar.update(batch_size)
                count += 1
                mean_outer_loss += (results['mean_outer_loss'] - mean_outer_loss) / count
                postfix = {'outer loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after']) - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

                accuracies.append(results['accuracies_after'])

                if verbose:
                    print("\ninner loss (labeled): \n{}".format(results['inner_losses_labeled']))
                    print("inner loss (unlabeled): \n{}".format(results['inner_losses_unlabeled']))
                    print("inner loss (labeled+unlabeled):\n{}".format(results['inner_losses']))

                    # print("inner loss stat_selected:\n{}".format(results['stat_selected']))

                for k, v in results.items():
                    result_per_epoch[k].append(v)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy


        metaval_accuracies = np.concatenate(accuracies, axis=0)
        # print(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(len(metaval_accuracies))
        # print(ci95)
        return mean_results, result_per_epoch, ci95


    def evaluate_iter(self, dataloader, max_batches=500, progress=0):
        num_batches = 0
        self.model.eval()    # key point todo: check this 10-17
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results, _ = self.get_outer_loss_evaluate(batch, self.num_adaptation_steps_test, progress=progress)
                yield results

                num_batches += 1
