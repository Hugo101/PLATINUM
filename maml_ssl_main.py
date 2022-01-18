import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'
import time
import json
import logging
import math
import pickle
import torch
from collections import defaultdict
from configuration import arg_parser
import common_tools as ct

from datasets_meta.dataloader_meta import BatchMetaDataLoader
from maml.datasets_benchmark import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning, ModelAgnosticMetaLearningBaseline, ModelAgnosticMetaLearningComb

args = arg_parser.parse_args()

ct.set_random_seeds(args.seed)

INTERVAL = 50
INTERVAL_VAL = args.interval_val

def cat_data(result_dict, new_dict):
    for key, values in new_dict.items():
        result_dict[key] += values
    return result_dict


def append_data(result_dict, new_dict):
    for key, values in new_dict.items():
        result_dict[key].append(values)
    return result_dict


def maml_ssl_main(args, device):
    benchmark = get_benchmark_by_name(args.dataset,
                                      args.data_folder,
                                      args.scenario,
                                      args.num_ways,
                                      args.num_shots,                     # shots in support set
                                      args.num_shots_test_meta_train,     # shots in query set for meta-train
                                      args.num_shots_test_meta_test,      # shots in query set for meta-test
                                      args.num_shots_unlabeled,   # num of unlabeled images for meta-train tasks
                                      args.num_shots_unlabeled_evaluate,  # num of unlabeled images per class
                                      args.num_classes_distractor,     # with distractor
                                      args.num_shots_distractor,       # with distractor
                                      args.num_shots_distractor_eval,  # with distractor
                                      args.num_unlabel_total,           # for "random"
                                      args.num_unlabel_total_evaluate,  # for "random"
                                      hidden_size=args.hidden_size)

    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=False)    # possible to avoid the leaking caffe2 warning

    meta_valid_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size_val,
                                              shuffle=True,    # make it to be false to debug
                                              num_workers=args.num_workers,
                                              pin_memory=False)

    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                              batch_size=args.batch_size_test,
                                              shuffle=True,  # make it to be false to debug
                                              num_workers=args.num_workers,
                                              pin_memory=False)

    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)

    # debugging
    metalearner = ModelAgnosticMetaLearningComb(benchmark.model,
                                            meta_optimizer,
                                            step_size=args.step_size,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            num_adaptation_steps_test=args.num_steps_evaluate,
                                            loss_function=benchmark.loss_function,
                                            coef_inner=args.coef_inner,
                                            coef_outer=args.coef_outer,
                                            device=device)

    # if args.ssl_algo == "SMI":
    #     metalearner = ModelAgnosticMetaLearning(benchmark.model,
    #                                             meta_optimizer,
    #                                             step_size=args.step_size,
    #                                             first_order=args.first_order,
    #                                             num_adaptation_steps=args.num_steps,
    #                                             num_adaptation_steps_test=args.num_steps_evaluate,
    #                                             loss_function=benchmark.loss_function,
    #                                             coef_inner=args.coef_inner,
    #                                             coef_outer=args.coef_outer,
    #                                             device=device)
    # else:
    #     metalearner = ModelAgnosticMetaLearningBaseline(benchmark.model,
    #                                             meta_optimizer,
    #                                             step_size=args.step_size,
    #                                             first_order=args.first_order,
    #                                             num_adaptation_steps=args.num_steps,
    #                                             num_adaptation_steps_test=args.num_steps_evaluate,
    #                                             loss_function=benchmark.loss_function,
    #                                             coef_inner=args.coef_inner,
    #                                             coef_outer=args.coef_outer,
    #                                             device=device)

    best_value = None

    real_datasets = ["miniimagenet", "omniglot", "tieredimagenet", "cifarfs"]
    results_train = defaultdict(list)  # store all results from meta-training
    results_valid = defaultdict(list)   # store all results from meta-validation
    results_test = defaultdict(list)
    results_mean_val_tst_epochs = {"mean_loss_val": [],
                                   "mean_accu_val": [],
                                   "mean_loss_tst": [],
                                   "mean_accu_tst": []}   # loss and accu of query set from meta-validation
    # Training loop
    epoch_desc_train = 'Epoch {{0: <{0}d}} (meta-train)'.format(1 + int(math.log10(args.num_epochs)))
    epoch_desc_val = 'Epoch {{0: <{0}d}} (meta-valid)'.format(1 + int(math.log10(args.num_epochs)))
    epoch_desc_tst = 'Epoch {{0: <{0}d}} (meta-test)'.format(1 + int(math.log10(args.num_epochs)))

    # # load the saved variables to resume the training
    # if args.resume:
    #     # saved results
    #     with open(path_resume+ "results_train.pkl", "rb") as f:
    #         results_train = pickle.load(f)
    #     with open(path_resume+ "results_valid.pkl", "rb") as f:
    #         results_valid = pickle.load(f)
    #     with open(path_resume+ "results_test.pkl", "rb") as f:
    #         results_test = pickle.load(f)
    #     with open(path_resume+ "results_mean_valid_test.json", "rb") as f:
    #         results_mean_val_tst_epochs = json.load(f)

    start_epoch = 0
    for epoch in range(start_epoch+1, args.num_epochs+1):
        # meta training
        result_train_per_epoch = metalearner.train(meta_train_dataloader,
                                             max_batches=args.num_batches,
                                             batch_size=args.batch_size,
                                             verbose=args.verbose,
                                             progress=epoch,
                                             desc=epoch_desc_train.format(epoch),
                                             )
        results_train = cat_data(results_train, result_train_per_epoch)


        if epoch % INTERVAL_VAL == 0:
            # meta validation
            results_mean_val, results_all_tasks_val = metalearner.evaluate(meta_valid_dataloader,
                                                                           max_batches=args.num_batches,
                                                                           batch_size=args.batch_size_val,
                                                                           verbose=args.verbose,
                                                                           progress=epoch,
                                                                           desc=epoch_desc_val.format(epoch))
            results_valid = append_data(results_valid, results_all_tasks_val)
            results_mean_val_tst_epochs["mean_loss_val"].append(results_mean_val['mean_outer_loss'])
            results_mean_val_tst_epochs["mean_accu_val"].append(results_mean_val["accuracies_after"])

            # meta test
            # results_mean_tst, results_all_tasks_tst = {}, {}
            if args.dataset in real_datasets:
                results_mean_tst, results_all_tasks_tst = metalearner.evaluate(meta_test_dataloader,
                                                                               max_batches=args.num_batches,
                                                                               batch_size=args.batch_size_test,
                                                                               verbose=args.verbose,
                                                                               progress=epoch,
                                                                               desc=epoch_desc_tst.format(epoch))
                results_test = append_data(results_test, results_all_tasks_tst)
                results_mean_val_tst_epochs["mean_loss_tst"].append(results_mean_tst['mean_outer_loss'])
                results_mean_val_tst_epochs["mean_accu_tst"].append(results_mean_tst["accuracies_after"])
            else:
                results_mean_tst, results_all_tasks_tst = {}, {}

            # save the validation acc and loss during each epoch
            rst_path_valid_test = os.path.abspath(os.path.join(args.output_subfolder, "results_mean_valid_test.json"))
            with open(rst_path_valid_test, "w") as f:
                json.dump(results_mean_val_tst_epochs, f, indent=2)

            # ### Save the best model based on validation set
            save_model = False
            if 'accuracies_after' in results_mean_val:
                if (best_value is None) or (best_value < results_mean_val['accuracies_after']):
                    best_value = results_mean_val['accuracies_after']
                    save_model = True
            elif (best_value is None) or (best_value > results_mean_val['mean_outer_loss']):
                best_value = results_mean_val['mean_outer_loss']
                save_model = True
            else:
                save_model = False

            if save_model and (args.output_folder is not None):
                print(f"^^^^^ Best model at Epoch: {epoch}")
                best_epoch={"epoch": epoch,
                            "valid": results_mean_val,
                            "test": results_mean_tst,
                            }
                with open(f"{args.model_path}.th", 'wb') as f:
                # with open(f"{args.model_path}_epoch_{epoch}.th", 'wb') as f:
                    torch.save(benchmark.model.state_dict(), f)
                with open(args.result_path, 'wb') as handle:
                    pickle.dump(best_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # ###

            # save model every interval_val epochs
            with open(f"{args.model_path}_epoch_{epoch}.th", 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

            # save some intemediate results, overwrite them epoch by epoch
            result_train_path = os.path.abspath(os.path.join(args.output_subfolder, f'results_train.pkl'))
            with open(result_train_path, "wb") as f:
                pickle.dump(results_train, f, protocol=pickle.HIGHEST_PROTOCOL)
            result_valid_path = os.path.abspath(os.path.join(args.output_subfolder, f"results_valid.pkl"))
            with open(result_valid_path, "wb") as f:
                pickle.dump(results_valid, f, protocol=pickle.HIGHEST_PROTOCOL)
            result_test_path = os.path.abspath(os.path.join(args.output_subfolder, f"results_test.pkl"))
            with open(result_test_path, "wb") as f:
                pickle.dump(results_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()


def main():
    start = time.time()  # float
    ct.create_path(args.output_folder)
    specific_file_name, tag = "", ""
    # base_path/ssl_path/N-way K-shot/specific_model
    # base model folder, storing all results of all experiments
    base_path = os.path.join(args.output_folder, f"{args.dataset}_{args.scenario}")
    # specific model folder, storing the specific model (specific combination in the configure file)
    if args.ssl_algo == "SMI":
        ssl_path = os.path.join(base_path, f"{args.ssl_algo}_{args.selection_option}_firstOrder_{args.first_order}")
    else:
        ssl_path = os.path.join(base_path, f"{args.ssl_algo}_firstOrder_{args.first_order}")
    # specific stopping policy model results
    few_shot_path = os.path.join(ssl_path, f"#way_{args.num_ways}_#shot_{args.num_shots}")

    if args.ssl_algo in ["SMI", "SMIcomb"]:
        tag = '_'.join(['BudgetS', str(args.budget_s), 'BudgetQ', str(args.budget_q), "TrueLabel", str(args.select_true_label)])
    elif args.ssl_algo == "PL":
        tag = '_'.join(['TH', str(args.pl_threshold), "TrueLabel", str(args.select_true_label)])
    elif args.ssl_algo in ["PLtopZ", "PLtopZperClass", "PLtopZperClassPLtopZ"]:
        tag = '_'.join(['TopZs', str(args.pl_num_topz), 'TopZq', str(args.pl_num_topz_outer), "TrueLabel", str(args.select_true_label)])

    if not args.resume:
        specific_file_name = time.strftime('%Y-%m-%d-%H%M%S') + "_" + tag + "_" + '_'.join(
            ['LabelRatio', str(args.ratio), '#ShotU', str(args.num_shots_unlabeled), '#InnerLR', str(args.step_size), 'Seed', str(args.seed), ])
        specific_model_path = os.path.join(few_shot_path, specific_file_name)
        ct.create_path(specific_model_path)
        args.output_subfolder = os.path.abspath(specific_model_path)   # absolute path
    else:
        args.output_subfolder = os.path.abspath(args.checkpoint_path)  # absolute path

    ct.set_logger('{}/log_file_outerLossAcc_seed_{}'.format(args.output_subfolder, args.seed), 'err')   # log file
    # ct.set_logger('{}/log_file_seed_{}'.format(args.output_subfolder, args.seed), 'out')   # log file
    print('Random Seed: {}'.format(args.seed))
    ct.set_random_seeds(args.seed)
    device = ct.set_device(args.gpu_id)

    args.model_path = os.path.abspath(os.path.join(args.output_subfolder, 'best_model'))
    args.result_path = os.path.abspath(os.path.join(args.output_subfolder, 'best_model_valid_test_result.pkl'))
    # save the config, json is better here because it is easily to open directly
    with open(os.path.join(args.output_subfolder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("****** Model parameters: \n{")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("} ****** \n")
    maml_ssl_main(args, device)
    time_used = "{:.2f}".format(time.time() - start)
    print(f"Total time used: {time_used}")
    with open(os.path.join(args.output_subfolder, 'others.json'), 'w') as fi:
        json.dump({"TimeUsed (s)": float(time_used)}, fi, indent=2)

if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.9, 0)
    main()
