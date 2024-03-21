# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from collections import defaultdict

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import wandb
from tqdm import tqdm

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    
    # changed default from None to 100
    parser.add_argument('--checkpoint_freq', type=int, default=300,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--save_linear_probed_clip', action = 'store_true')
    parser.add_argument('--sweep', type=bool, default=False)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    # remove the single quotes from the output_dir
    args.output_dir = args.output_dir.replace("'", "")
    #print('path:', args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    print("\tTesting environment: {}".format(args.test_envs))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.sweep:
        print('Not saving models')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
    if args.algorithm in ['SCMD_focal2']:
        temp_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(in_splits)
            if i not in args.test_envs]
        num_classes = dataset.num_classes
        class_counts = defaultdict(int)
        total_count = 0

        for loader in temp_loaders:
            for _, targets in loader:
                for target in targets:
                    class_counts[target.item()] += 1
                    total_count += 1
        weights = {class_id: total_count / count for class_id, count in class_counts.items()}
        hparams['alpha'] = torch.tensor([weights[i] for i in range(num_classes)], requires_grad=False).to(device)
        print(hparams['alpha'])
            
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]



    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)


    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    print('n_steps', n_steps)
    #checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    checkpoint_freq = dataset.CHECKPOINT_FREQ
    print('checkpoint_freq', checkpoint_freq)

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    # --------------------
    # WKD and W2D requires step
    add_step = False
    if args.algorithm in ['WKD', 'W2D', 'SCMD', 'Baseline', 'SCMD_iid', 'BSS', 'SCMD_no_CM', 'SCMD_focal', 'SCMD_focal2']:
        add_step = True

    last_results_keys = None
    #------------------------
    best_acc = float('-inf')
    # if not args.sweep:
    #     name = args.algorithm + '_' + args.dataset + '_' + str(args.seed)
    #     wandb.init(project='DG_Research', entity='seanl', name = name, job_type = 'train')
    #     wandb.config.update(args)
    #------------------------
    step_times = []
    for step in tqdm(range(start_step, n_steps), file=sys.__stdout__, leave=False):
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]

        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        
        step_start_time = time.time()
        if add_step:
            step_vals = algorithm.update(minibatches_device, uda_device, step)
        else:
            step_vals = algorithm.update(minibatches_device, uda_device)

        if step != 0:
            step_times.append(time.time() - step_start_time)
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
        else:
            checkpoint_vals['step_time'].append(0)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            #------------------------------------------------------------------------------------------
            # SMA
            agg_val_acc, nagg_val_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and 'out' in name and int(name.split('env')[1].split('_')[0]) not in args.test_envs:
                    agg_val_acc += results[name]
                    nagg_val_acc += 1.
            agg_val_acc /= (nagg_val_acc + 1e-9)
            results['agg_val_acc'] = agg_val_acc

            agg_test_acc, nagg_test_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and name !='agg_val_acc' and int(name.split('env')[1].split('_')[0]) in args.test_envs:
                    agg_test_acc += results[name]
                    nagg_test_acc += 1.
            agg_test_acc /= (nagg_test_acc + 1e-9)
            results['agg_test_acc'] = agg_test_acc
            #------------------------------------------------------------------------------------------

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            # if not args.sweep:
            #     wandb.log(results)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            # if best_acc < agg_val_acc:
            #     print("Saving best model...")
            #     best_acc = agg_val_acc
            #     save_checkpoint('best_model.pkl')
    
    print('Average step time:', np.mean(step_times))
    last_k_epoch = hparams.get('last_k_epoch', 0)
    full_data_step = int(5000 * (1 - last_k_epoch))
    print('Average step time before full data:', np.mean(step_times[:full_data_step]))
    print('Average step time after full data:', np.mean(step_times[full_data_step:]))
    np.save(os.path.join(args.output_dir, 'step_times.npy'), step_times)
    

    if args.visualize:
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        print('Visualizing')
        for name, loader, weights in evals:
            misc.visualize(algorithm, loader, device)




    if not args.sweep:
        save_checkpoint('model.pkl')

    if args.save_linear_probed_clip:
        torch.save(algorithm.classifier.state_dict(), 'pretrained_models/clip_classifier.pth')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    # wandb.finish()


