import os
import time
import json
import logging
import math
import pickle
import torch
from configuration import arg_parser
import common_tools as ct

from datasets_meta.dataloader_meta import BatchMetaDataLoader
from maml.datasets_benchmark import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

args = arg_parser.parse_args()

INTERVAL = 50
INTERVAL_VAL = 1

def maml_ssl_test(args, device):
    benchmark = get_benchmark_by_name(args.dataset,
                                      args.data_folder,
                                      args.scenario,               #
                                      args.num_unlabel_total,           # for "random"
                                      args.num_unlabel_total_evaluate,  # for "random"
                                      args.num_ways,
                                      args.num_shots,             # shots in support set
                                      args.num_shots_test,        # shots in query set
                                      args.num_shots_unlabeled,   # num of unlabeled images for meta-train tasks
                                      args.num_shots_unlabeled_evaluate,  # num of unlabeled images per class
                                      hidden_size=args.hidden_size)

    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                              batch_size=args.batch_size_test,
                                              shuffle=True,  # make it to be false to debug
                                              num_workers=args.num_workers,
                                              pin_memory=False)

    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)

    pretrain_model_path = "/data/cxl173430/MAML_with_SSL_Git/output/2021-11-03_011310_wo_distractor_miniimagenet_SMI_0.1_Shot_1_CoefU_-1.0_#ShotU_10_Seed_123_Budget_25_TrueLabel_False/"
    # pretrain_model_path = "/data/cxl173430/MAML_with_SSL_Git/output/2021-11-03_011435_wo_distractor_miniimagenet_SMI_0.4_Shot_1_CoefU_-1.0_#ShotU_10_Seed_123_Budget_25_TrueLabel_False/"

    model_name = "best_model.th"
    stat_load = torch.load(pretrain_model_path+model_name)
    benchmark.model.load_state_dict(stat_load)

    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            step_size=args.step_size,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            num_adaptation_steps_test=args.num_steps_evaluate,
                                            loss_function=benchmark.loss_function,
                                            coef=args.coef,
                                            device=device,
                                            progress=0)  # progress is added only for tuning coef

    # meta test
    results_mean_tst, results_all_tasks_tst = {}, {}
    results_mean_tst, results_all_tasks_tst = metalearner.evaluate(meta_test_dataloader,
                                                                   max_batches=args.num_batches,
                                                                   batch_size=args.batch_size_test,
                                                                   verbose=args.verbose,
                                                                   progress=300,
                                                                   desc="meta-test")
    print("results_mean_tstL: ", results_mean_tst)
    print("results_all_tasks_tst: ", results_all_tasks_tst)
    # results_test.append(results_all_tasks_tst)
    # results_mean_val_tst_epochs["mean_loss_tst"].append(results_mean_tst['mean_outer_loss'])
    # results_mean_val_tst_epochs["mean_accu_tst"].append(results_mean_tst["accuracies_after"])
    #
    # # save the validation acc and loss during each epoch
    # rst_path_valid_test = os.path.abspath(os.path.join(args.output_subfolder, "results_mean_valid_test.json"))
    # with open(rst_path_valid_test, "w") as f:
    #     json.dump(results_mean_val_tst_epochs, f, indent=2)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()


def main():
    ct.create_path(args.output_folder)
    file_name, tag = "", ""
    if args.scenario == "woDistractor":
        if args.ssl_algo == "SMI":
            tag = '_'.join(['Budget', str(args.budget), "TrueLabel", str(args.select_true_label)])
        elif args.ssl_algo == "PL":
            tag = '_'.join(['TH', str(args.pl_threshold), "TrueLabel", str(args.select_true_label)])
        elif args.ssl_algo == "PL_topZ":
            tag = '_'.join(['TopZ', str(args.pl_topz), "TrueLabel", str(args.select_true_label)])

        file_name = time.strftime('%Y-%m-%d_%H%M%S') + "_" + \
                    '_'.join([args.scenario, args.dataset, args.ssl_algo, str(args.ratio),
                              'Shot', str(args.num_shots),
                              'CoefU', str(args.coef),
                              '#ShotU', str(args.num_shots_unlabeled),
                              'Seed', str(args.seed),
                              ]) + "_" + tag

    elif args.scenario == "random" or args.scenario == "all_ood":
        if args.ssl_algo == "SMI":
            tag = '_'.join(['Budget', str(args.budget)])
        elif args.ssl_algo == "PL":
            tag = '_'.join(['TH', str(args.pl_threshold)])
        elif args.ssl_algo == "PL_topZ":
            tag = '_'.join(['TopZ', str(args.pl_topz)])

        file_name = time.strftime('%Y-%m-%d_%H%M%S') + "_" + \
                    '_'.join([args.scenario, args.dataset, args.ssl_algo, str(args.ratio),
                              'Shot', str(args.num_shots),
                              'CoefU', str(args.coef),
                              'Total#U', str(args.num_unlabel_total),
                              'Seed', str(args.seed),
                              ]) + "_" + tag

    output_subfolder = os.path.join(args.output_folder, file_name)
    ct.create_path(output_subfolder)
    args.output_subfolder = os.path.abspath(output_subfolder)   # absolute path
    ct.set_logger('{}/log_file_outerLossAcc_seed_{}'.format(args.output_subfolder, args.seed), 'err')   # log file
    # ct.set_logger('{}/log_file_seed_{}'.format(args.output_subfolder, args.seed), 'out')   # log file

    print('Random Seed: {}'.format(args.seed))
    ct.set_random_seeds(args.seed)
    device = ct.set_device(args.gpu_id)

    args.model_path = os.path.abspath(os.path.join(args.output_subfolder, 'best_model'))
    args.result_path = os.path.abspath(os.path.join(args.output_subfolder, 'best_model_valid_test_result.pkl'))
    with open(os.path.join(args.output_subfolder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("****** Model parameters: \n{")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("} ****** \n")
    maml_ssl_test(args, device)


if __name__ == "__main__":
    start = time.time()   # float
    main()
    print("Total time used: {:.2f}".format(time.time() - start))
