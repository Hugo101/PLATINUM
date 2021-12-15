import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
import time
from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from maml.utils import tensors_to_device, compute_accuracy, model_update
from lib_SSL.config import config
import pickle
from configuration import arg_parser

args = arg_parser.parse_args()

# __all__ = ['ModelAgnosticMetaLearningBaseline', 'MAML', 'FOMAML']
__all__ = ['ModelAgnosticMetaLearningBaseline']

if args.ssl_algo == 'VAT':
    print("SSL algorithm: VAT")
    from lib_SSL.algs.vat import VAT
    alg_cfg = config['VAT']
    consis_coef = alg_cfg['consis_coef']
    ssl_obj = VAT(alg_cfg["eps"]["svhn"], alg_cfg["xi"], 1)

if args.ssl_algo == 'PL':
    print(f"##### SSL algorithm: PL (threshold), TrueLabel:{args.select_true_label}")
    from lib_SSL.algs.pseudo_label import PL
    alg_cfg = config['PL']
    consis_coef = alg_cfg['consis_coef']

    ssl_obj       = PL(args.num_ways, args.pl_threshold, args.pl_batch_size, args.select_true_label, args.scenario, args.verbose)
    ssl_obj_outer = PL(args.num_ways, args.pl_threshold_outer, args.pl_batch_size, args.select_true_label, args.scenario, args.verbose)
    # ssl_obj_joint = PL(args.num_ways, args.pl_threshold, args.select_true_label, args.scenario, args.verbose, support_unlabeled_avg_flag=True)

if args.ssl_algo == 'PLtopZ':
    from lib_SSL.algs.pseudo_label_topZ import PLtopZ
    consis_coef = 1

    print(f"## SSL algorithm (inner loop): PL with Top Z (TH:{args.pl_threshold}), TrueLabel:{args.select_true_label}")
    ssl_obj       = PLtopZ(args.num_ways, args.pl_num_topz, args.pl_batch_size, args.select_true_label, args.scenario, args.verbose, args.pl_threshold)

    print(f"## SSL algorithm (outer loop): PL with Top Z (TH:{args.pl_threshold_outer}), TrueLabel:{args.select_true_label}")
    ssl_obj_outer = PLtopZ(args.num_ways, args.pl_num_topz_outer, args.pl_batch_size, args.select_true_label, args.scenario, args.verbose, args.pl_threshold_outer)

consis_coef_outer = 1
WARMSTART_EPOCH = 100
WARMSTART_ITER = 10000
WARM = 0

class ModelAgnosticMetaLearningBaseline(object):
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1,
                 num_adaptation_steps_test=10,
                 scheduler=None,
                 loss_function=F.cross_entropy,
                 coef_inner=0.01,
                 coef_outer=0.01,
                 device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.num_adaptation_steps_test = num_adaptation_steps_test
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.coef_inner = coef_inner  # added
        self.coef_outer = coef_outer  # added
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
                "coef_inner": [],
                "coef_outer": [],
                'ratio_ood_selected': [],
                # store inner and outer selection accuracy
                'accuracy_selected': [],
                "accuracy_change_per_task": [],
                "accuracies_after": [],
        }
        with tqdm(total=max_batches*batch_size, disable=False, **kwargs) as pbar:
            batch_id = 1
            for results in self.train_iter(dataloader, max_batches=max_batches, progress=progress):
                pbar.update(batch_size)

                postfix = {'outer loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy (query)'] = '{0:.4f}'.format(np.mean(results['accuracies_after']))

                pbar.set_postfix(**postfix)

                if verbose:
                    print(f"\niteration: {progress*batch_id}, outer_SSL_coef: {results['coef_outer']}")
                    print("inner loss (labeled): \n{}".format(results['inner_losses_labeled']))
                    print("inner loss (unlabeled): \n{}".format(results['inner_losses_unlabeled']))
                    print("inner loss (labeled+unlabeled): \n{}".format(results['inner_losses']))
                    print("inner and outer ratio of ood selection:\n{}".format(results['ratio_ood_selected']))
                    print("inner and outer accuracy_selected:\n{}".format(results['accuracy_selected']))
                    print("coeffcient in the inner loop:\n{}\n".format(results['coef_inner']))

                for k,v in results.items():
                    result_per_epoch[k].append(v)

                batch_id += 1

        return result_per_epoch


    def train_iter(self, dataloader, max_batches=500, progress=0):
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

                batch = tensors_to_device(batch, device=self.device)
                torch.cuda.empty_cache()
                outer_loss, results = self.get_outer_loss(batch, self.num_adaptation_steps,
                                                          progress=progress,
                                                          sub_progress=num_batches)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1


    def get_outer_loss(self, batch, num_adapt_steps, progress=0, sub_progress=0):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, query_targets = batch['test']
        num_tasks = query_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses_labeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses_unlabeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'coef_inner': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'outer_losses_unlabeled': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'coef_outer': 0.,
            'ratio_ood_selected': np.empty((num_adapt_steps+1, num_tasks), dtype=object), # include outer loop
            # store inner and outer selection accuracy
            'accuracy_selected': np.empty((num_adapt_steps+1, num_tasks), dtype=object), # include outer loop
            "accuracy_change_per_task": np.zeros((2, num_tasks), dtype=np.float32),
            # accu before inner loop
            # accu of support after inner loop
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32),  # accu of query set after outer loop
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (support_inputs, support_targets, query_inputs, query_targets, unlabeled_inputs, unlabeled_targets) \
                in enumerate(zip(*batch['train'], *batch['test'], *batch['unlabeled'])):
            # print(f"task_id:{task_id}, unlabeled targets: {unlabeled_targets}")
            # inner loop
            params, adaptation_results, selected_ids_inner_set = self.adapt(support_inputs, support_targets,
                                                    unlabeled_inputs, unlabeled_targets,
                                                    is_classification_task=True,
                                                    num_adaptation_steps=num_adapt_steps,
                                                    step_size=self.step_size,
                                                    first_order=self.first_order,
                                                    meta_train=True,
                                                    coef=self.coef_inner,
                                                    progress=progress, sub_progress=sub_progress)

            results['inner_losses_labeled'][:, task_id] = adaptation_results['inner_losses_labeled']
            results['inner_losses_unlabeled'][:, task_id] = adaptation_results['inner_losses_unlabeled']
            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results['coef_inner'][:, task_id] = adaptation_results['coef_inner']
            results["accuracy_change_per_task"][0, task_id] = adaptation_results['accuracy_before']
            results["accuracy_change_per_task"][1, task_id] = adaptation_results['accuracy_support']

            # outer loop
            with torch.set_grad_enabled(self.model.training):
                query_logits = self.model(query_inputs, params=params)
                outer_loss = self.loss_function(query_logits, query_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss
                results["accuracies_after"][task_id] = compute_accuracy(query_logits, query_targets)

            # outer subset selection
            print("\n++++++ outer loop:") if args.verbose else None
            if not args.is_outer_selection:
                loss_unlabeled, ratio_ood, acc_slct = 0, 0, 0
            else:
                torch.cuda.empty_cache()
                outputs_unlabeled = self.model(unlabeled_inputs, params=params)
                torch.cuda.empty_cache()
                loss_unlabeled, ratio_ood, acc_slct, selected_ids = ssl_obj_outer(unlabeled_inputs,
                                                                            outputs_unlabeled.detach(),
                                                                            self.model, params,
                                                                            unlabeled_targets,
                                                                            selected_ids_inner_set)

            print(f"******** TaskID:{task_id}, outloop SMI selection accuracy: {acc_slct}\n") if args.verbose else None
            if type(loss_unlabeled) == torch.Tensor:
                results['outer_losses_unlabeled'][task_id] = loss_unlabeled.item()
            else:
                results['outer_losses_unlabeled'][task_id] = loss_unlabeled
            adaptation_results['ratio_ood_selected'].append(ratio_ood)
            adaptation_results['accuracy_selected'].append(acc_slct)
            results['ratio_ood_selected'][:, task_id] = adaptation_results['ratio_ood_selected']
            results['accuracy_selected'][:, task_id] = adaptation_results['accuracy_selected']

            if self.coef_outer >= 0:
                coeff = self.coef_outer
            elif self.coef_outer == -1:     # epoch
                coeff = consis_coef_outer * math.exp(-5 * (1 - min(progress / WARMSTART_EPOCH, 1)) ** 2)
            elif self.coef_outer == -2:     # iteration
                coeff = consis_coef_outer * math.exp(-5 * (1 - min((progress * sub_progress) / WARMSTART_ITER, 1)) ** 2)
            mean_outer_loss += coeff*loss_unlabeled

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()
        results['coef_outer'] = coeff

        return mean_outer_loss, results


    def adapt(self, inputs, targets,
              unlabeled_inputs, unlabeled_targets,
              is_classification_task=True,
              num_adaptation_steps=1,
              step_size=0.1,
              first_order=False,
              meta_train=True,
              coef=0.01,                     # ssl
              progress=0, sub_progress=0):

        params = None
        results = {'inner_losses_labeled': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'inner_losses_unlabeled': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'inner_losses': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'coef_inner': np.zeros(num_adaptation_steps, dtype=np.float32),
                   'ratio_ood_selected': [],
                   'accuracy_selected': [],
                   }
        selected_ids_inner = []

        for step in range(num_adaptation_steps):
            print(f"\n++++++ At Inner Step {step+1}:") if args.verbose else None
            outputs_labeled = self.model(inputs, params=params)
            # torch.cuda.empty_cache()
            # print("################# sleeping for 60 secs")
            # time.sleep(60)
            # print("################# wake up")
            if self.support_unlabeled_avg:
                print("++++++++++++ Calculate the mean loss of support set and selected unlabeled set.")
                # losses_unlabeled, num_UN_SLT = ssl_obj_joint(inputs_labeled_unlabeled,
                #                                              outputs_labeled_unlabeled.detach(),
                #                                              self.model, params,
                #                                              unlabeled_mask, targets_labeled_unlabeled)
                # assert torch.count_nonzero(losses_unlabeled).item() == num_UN_SLT
                #
                # losses_support = self.loss_function(outputs_labeled_unlabeled, target, reduction="none",
                #                                     ignore_index=-1)
                #
                # num_support_selected = torch.count_nonzero(losses_unlabeled).item() + \
                #                        torch.count_nonzero(1 - unlabeled_mask).item()
                # print(
                #     f"++++++ At Step {step}: support loss vs. pseudo loss: {torch.sum(losses_support)} vs. {torch.sum(losses_unlabeled)}")
                # print(f"++++++++++++ The size of    support set: {torch.count_nonzero(1 - unlabeled_mask).item()}")
                # print(f"++++++++++++ The size of SLTed UnLB set: {torch.count_nonzero(losses_unlabeled).item()}")
                # print(f"++++++++++++ The size of support set + SLTed UnLB set: {num_support_selected}")
                #
                # loss_support = torch.sum(losses_support) / num_support_selected
                # loss_unlabeled = torch.sum(losses_unlabeled) / num_support_selected
                # inner_loss = loss_support + loss_unlabeled
                # results['inner_losses_labeled'][step] = loss_support.item()
                # results['inner_losses_unlabeled'][step] = loss_unlabeled
                # results['inner_losses'][step] = inner_loss
            else:
                # # SSL loss
                if coef >= 0:
                    coeff = coef
                elif coef == -1:  # num_adaptation_steps
                    coeff = consis_coef * math.exp(-5 * (1 - min(step / num_adaptation_steps, 1)) ** 2)
                print(f"==============++++++++ coeff:{coeff}") if args.verbose else None
                results['coef_inner'][step] = coeff

                if args.ssl_algo == "PLtopZ" or args.ssl_algo == "PL" or args.ssl_algo == "VAT":
                    outputs_unlabeled = self.model(unlabeled_inputs, params=params)
                    loss_unlabeled, ratio_ood, acc_slct, selected_ids = ssl_obj(unlabeled_inputs,
                                                                  outputs_unlabeled.detach(),
                                                                  self.model, params,
                                                                  unlabeled_targets)
                    selected_ids_inner.extend(selected_ids)

                # several POC setting
                elif args.ssl_algo == "mamlLargeQ" or args.ssl_algo == "mamlLargeS" \
                        or args.ssl_algo == "mamlLargeSandQ" or args.ssl_algo == "largeSandQDiff" \
                        or args.ssl_algo == "MAML":
                    loss_unlabeled, ratio_ood, acc_slct = 0, 0, 0

                torch.cuda.empty_cache()
                # # Supervised Loss
                loss_support = self.loss_function(outputs_labeled, targets, reduction="mean")
                inner_loss = loss_support + loss_unlabeled * coeff
                results['inner_losses_labeled'][step] = loss_support.item()
                results['inner_losses_unlabeled'][step] = loss_unlabeled
                results['inner_losses'][step] = inner_loss
                results['ratio_ood_selected'].append(ratio_ood) if args.scenario == "wDistributes" else None
                results['accuracy_selected'].append(acc_slct)

            if (step == 0) and is_classification_task:
                # acc before inner loop training 10-14-2021
                results['accuracy_before'] = compute_accuracy(outputs_labeled, targets)

            if (step == num_adaptation_steps-1) and is_classification_task:
                # acc for support set after inner loop training
                results['accuracy_support'] = compute_accuracy(outputs_labeled, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                                                step_size=step_size, params=params,
                                                first_order=(not self.model.training) or first_order)

        if args.ssl_algo == "SMI" or args.ssl_algo == "PLtopZ":
            return params, results, set(selected_ids_inner)

        return params, results, None


    def get_outer_loss_evaluate(self, batch, num_adapt_steps, progress=0):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, query_targets = batch['test']
        num_tasks = query_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses_labeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses_unlabeled': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'coef_inner': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'inner_losses': np.zeros((num_adapt_steps, num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            # 'outer_losses_unlabeled': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'ratio_ood_selected': np.empty((num_adapt_steps, num_tasks), dtype=object),
            'accuracy_selected': np.empty((num_adapt_steps, num_tasks), dtype=object),
            "accuracy_change_per_task": np.zeros((2, num_tasks), dtype=np.float32),
            # accu before inner loop
            # accu of support after inner loop
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32),     # accu of query set after outer loop
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (support_inputs, support_targets, query_inputs, query_targets, unlabeled_inputs, unlabeled_targets) \
                in enumerate(zip(*batch['train'], *batch['test'], *batch['unlabeled'])):
            params, adaptation_results, _ = self.adapt(support_inputs, support_targets,
                                                    unlabeled_inputs, unlabeled_targets,
                                                    is_classification_task=True,
                                                    num_adaptation_steps=num_adapt_steps, step_size=self.step_size,
                                                    first_order=self.first_order,
                                                    meta_train=False,
                                                    coef=self.coef_inner,
                                                    progress=progress)

            results['inner_losses_labeled'][:, task_id] = adaptation_results['inner_losses_labeled']
            results['inner_losses_unlabeled'][:, task_id] = adaptation_results['inner_losses_unlabeled']
            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results['coef_inner'][:, task_id] = adaptation_results['coef_inner']
            results['accuracy_selected'][:, task_id] = adaptation_results['accuracy_selected']
            if args.scenario == "random":
                results['ratio_ood_selected'][:, task_id] = adaptation_results['ratio_ood_selected']

            results['accuracy_change_per_task'][0, task_id]  = adaptation_results['accuracy_before']
            results['accuracy_change_per_task'][1, task_id] = adaptation_results['accuracy_support']

            with torch.set_grad_enabled(self.model.training):    # For query set. meta-train: True, meta-val/test:False
                test_logits = self.model(query_inputs, params=params)
                outer_loss = self.loss_function(test_logits, query_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            results['accuracies_after'][task_id] = compute_accuracy(test_logits, query_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results


    def evaluate(self, dataloader, max_batches=500, batch_size=4, verbose=True, progress=0, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        result_per_epoch = {
            'num_tasks': [],
            'inner_losses_labeled': [],
            'inner_losses_unlabeled': [],
            'coef_inner': [],
            'inner_losses': [],
            'outer_losses': [],
            'mean_outer_loss': [],
            'ratio_ood_selected': [],
            # store inner and outer selection accuracy
            'accuracy_selected': [],
            "accuracy_change_per_task": [],
            "accuracies_after": [],
        }
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

                if verbose:
                    print("\ninner loss (labeled): \n{}".format(results['inner_losses_labeled']))
                    print("inner loss (unlabeled): \n{}".format(results['inner_losses_unlabeled']))
                    print("inner loss (labeled+unlabeled):\n{}".format(results['inner_losses']))
                    print("inner loss ratio of ood selection:\n{}".format(results['ratio_ood_selected']))
                    print("inner loss accuracy_selected:\n{}".format(results['accuracy_selected']))
                    print("coeffcient in the inner loop:\n{}\n".format(results['coef_inner']))

                for k, v in results.items():
                    result_per_epoch[k].append(v)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results, result_per_epoch


    def evaluate_iter(self, dataloader, max_batches=500, progress=0):
        num_batches = 0
        self.model.eval()    # key point todo: check this 10-17
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss_evaluate(batch, self.num_adaptation_steps_test, progress=progress)
                yield results

                num_batches += 1

#
# MAML = ModelAgnosticMetaLearning
#
#
# class FOMAML(ModelAgnosticMetaLearning):
#     def __init__(self, model, optimizer=None, step_size=0.1,
#                  learn_step_size=False, per_param_step_size=False,
#                  num_adaptation_steps=1, scheduler=None,
#                  loss_function=F.cross_entropy, device=None):
#         super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
#                                      step_size=step_size, learn_step_size=learn_step_size,
#                                      per_param_step_size=per_param_step_size,
#                                      num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
#                                      loss_function=loss_function, device=device)
