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

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import matplotlib.pyplot as plt
from cuml.manifold import TSNE
from tqdm import tqdm
import warnings


class ExtendedConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super(ExtendedConcatDataset, self).__init__(datasets)
        # Assuming all datasets have the same set of classes
        self.classes = datasets[0].classes
        self.train_labels = datasets[0].train_labels
        
if __name__ == "__main__":
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
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--algorithm_dict_path', type=str, default="algorithm_dict")
    args = parser.parse_args()

    # remove warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # # change output file name
    # store_prefix = f"{args.dataset}_{args.imb_type}_{args.imb_factor}" if 'Imbalance' in args.dataset else args.dataset
    # args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    # if args.test_envs is not None:
    #     args.store_name = f"{args.store_name}_env{str(args.test_envs).replace(' ', '')[1:-1]}"

    # misc.prepare_folders(args)
    # args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)

    args.output_dir = args.output_dir.replace("'", "")
    os.makedirs(args.output_dir, exist_ok=True)
    args.algorithm_dict_path = args.output_dir + '/' + args.algorithm_dict_path
    algorithms_dict = args.algorithm_dict_path

    print(f"===> algorithm_dict_path: {algorithms_dict}")


    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

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
    

    in_splits = []
    out_splits = []
    uda_splits = []
    test_splits = []

    # uda is 0 by default
    # 
    # with open("experiment/config.yaml", 'r') as stream:
    #     try:
    #         config = yaml.safe_load(stream)
    #         move_classes_from_train_to_val_test = config['move_classes']
    #     except yaml.YAMLError as exc:
    #         tqdm.write(exc)

    # if move_classes_from_train_to_val_test is not None:
    #     tqdm.write(f"===> Move classes from train to val and test: {move_classes_from_train_to_val_test}")

    """
    When we sweep, we can not specify the move config, so we need to handle the move classes logic here
    We do not remove any classes from the test environments, for the training environments, we remove equal number of classes from each environment
    """
    label_to_name = dataset.label_to_name
    all_classes = list(range(dataset.num_classes))
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
    
    tsne_loaders = [FastDataLoader(
        dataset=env,
        batch_size=256,
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits + out_splits + uda_splits)
        if i in args.test_envs]
    
    print('===============>', len(tsne_loaders))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    
    assert algorithms_dict is not None and os.path.isfile(algorithms_dict), "algorithm_dict_path is not valid"
    # load the algorithm_dict
    algorithm.load_state_dict(torch.load(algorithms_dict)['model_dict'])
    algorithm.to(device)

    # start tsne
    print(f"===> Start tsne")
    tsne = TSNE(n_components=2, init='pca', random_state = 1234, n_iter = 500, square_distances = True)
    feature_list = []
    label_list = []
    algorithm.network_sma.eval()
    for loader in tsne_loaders:
        for x, y in tqdm(loader, total=len(loader)):
            x = x.to(device)
            with torch.no_grad():
                features = algorithm.network_sma[0](x)
                features = algorithm.feature_projector(features)
                #features = algorithm.teacher(x)
            features = features.view(features.size(0), -1).cpu().numpy()
            feature_list.append(features)
            label_list.append(y.cpu().numpy())

    # stack all features and labels
    all_features = np.vstack(feature_list)
    all_labels = np.hstack(label_list)
    tsne_embedding = tsne.fit_transform(all_features)
    print(f"===> Finish tsne")

    # plot tsne
    print(f"===> Start plotting")


    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # # 假设 all_labels 是您的标签数组，tsne_embedding 是 t-SNE 降维后的结果
    # # 下面创建一个 DataFrame 来存储 t-SNE 的输出和对应的标签
    # import pandas as pd

    # # 首先，我们将 t-SNE 结果和标签合并到一个 DataFrame 中
    # df_tsne = pd.DataFrame(tsne_embedding, columns=['Dim1', 'Dim2'])
    # df_tsne['label'] = all_labels

    # # 使用 Seaborn 的 sns.scatterplot 绘制散点图
    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(
    #     x="Dim1", y="Dim2",
    #     hue="label", # 这里的 'label' 对应于上面 DataFrame 中的标签列
    #     palette=sns.color_palette("hsv", len(np.unique(all_labels))), # 使用 HSV 颜色空间中的色调
    #     data=df_tsne,
    #     legend="full",
    #     alpha=0.6
    # )

    # # 如果标签很多，可能需要调整图例的显示
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title('t-SNE visualization')
    # plt.savefig(os.path.join(args.output_dir, 'tsne.png'))
    # plt.show()


    plt.figure(figsize=(10, 10))

    unique_labels = np.unique(all_labels)
    scatter_plots = []

    for label in unique_labels:
        idxs = np.where(all_labels == label)
        scatter = plt.scatter(tsne_embedding[idxs, 0], tsne_embedding[idxs, 1], label=label_to_name[label], cmap='tab10')
        scatter_plots.append(scatter)

    # Display legend
    plt.legend(handles=scatter_plots, title="Classes")
    plt.savefig(os.path.join(args.output_dir, 'tsne.png'))
    print(f"===> Finish plotting")