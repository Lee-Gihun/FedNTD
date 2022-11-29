import torch
from torch.utils.data import DataLoader
from collections import Counter

import random
import numpy as np
import os

from .mnist.loader import get_all_targets_mnist, get_dataloader_mnist
from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10
from .cifar100.loader import get_all_targets_cifar100, get_dataloader_cifar100
from .cinic10.loader import get_all_targets_cinic10, get_dataloader_cinic10
from .tinyimagenet.loader import (
    get_all_targets_tinyimagenet,
    get_dataloader_tinyimagenet,
)

__all__ = ["data_distributer"]

DATA_INSTANCES = {
    "mnist": get_all_targets_mnist,
    "cifar10": get_all_targets_cifar10,
    "cifar100": get_all_targets_cifar100,
    "cinic10": get_all_targets_cinic10,
    "tinyimagenet": get_all_targets_tinyimagenet,
}
DATA_LOADERS = {
    "mnist": get_dataloader_mnist,
    "cifar10": get_dataloader_cifar10,
    "cifar100": get_dataloader_cifar100,
    "cinic10": get_dataloader_cinic10,
    "tinyimagenet": get_dataloader_tinyimagenet,
}


def data_distributer(
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    oracle_size=0,
    oracle_batch_size=None,
):
    """
    Distribute dataloaders for server and locals by the given partition method.
    """

    root = os.path.join(root, dataset_name)
    all_targets = DATA_INSTANCES[dataset_name](root)
    num_classes = len(np.unique(all_targets))
    net_dataidx_map_test = None

    local_loaders = {
        i: {"datasize": 0, "train": None, "test": None} for i in range(n_clients)
    }

    if partition.method == "centralized":
        net_dataidx_map = centralized_partition(all_targets)
    elif partition.method == "iid":
        net_dataidx_map = iid_partition(all_targets, n_clients)
    elif partition.method == "sharding":
        net_dataidx_map, rand_set_all = sharding_partition(
            all_targets, n_clients, partition.shard_per_user
        )
        all_targets_test = DATA_INSTANCES[dataset_name](root, train=False)
        net_dataidx_map_test, _ = sharding_partition(
            all_targets_test,
            n_clients,
            partition.shard_per_user,
            rand_set_all=rand_set_all,
        )
    elif partition.method == "sharding_max":
        net_dataidx_map = sharding_max_partition(all_targets, n_clients, partition.K)
    elif partition.method == "lda":
        net_dataidx_map = lda_partition(all_targets, n_clients, partition.alpha)
    else:
        raise NotImplementedError

    print(">>> Distributing client train data...")
    for client_idx, dataidxs in net_dataidx_map.items():
        local_loaders[client_idx]["datasize"] = len(dataidxs)
        local_loaders[client_idx]["train"] = DATA_LOADERS[dataset_name](
            root, train=True, batch_size=batch_size, dataidxs=dataidxs,
        )

    print(">>> Distributing client test data...")
    if net_dataidx_map_test is not None:
        for client_idx, dataidxs in net_dataidx_map_test.items():
            local_testloader = DATA_LOADERS[dataset_name](
                root, train=False, batch_size=batch_size, dataidxs=dataidxs,
            )
            local_loaders[client_idx]["test"] = local_testloader
            local_loaders[client_idx]["dist"] = get_dist_vec(
                local_testloader, num_classes
            )

    global_loaders = {
        "train": DATA_LOADERS[dataset_name](root, train=True, batch_size=batch_size),
        "test": DATA_LOADERS[dataset_name](root, train=False, batch_size=batch_size),
    }

    # Count class samples in Clients
    data_map = net_dataidx_map_counter(net_dataidx_map, all_targets)

    data_distributed = {
        "global": global_loaders,
        "local": local_loaders,
        "data_map": data_map,
        "num_classes": num_classes,
    }

    # Set oracle loader for CL-like memory
    oracle_idxs = oracle_partition(all_targets, oracle_size=oracle_size)
    obs = batch_size

    if oracle_batch_size is not None:
        obs = oracle_batch_size

    if oracle_idxs is not None:
        data_distributed["oracle"] = DATA_LOADERS[dataset_name](
            root, train=True, batch_size=obs, dataidxs=oracle_idxs
        )

    return data_distributed


def centralized_partition(all_targets):
    labels = all_targets
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    tot_idx = np.array(tot_idx)
    np.random.shuffle(tot_idx)
    net_dataidx_map[0] = tot_idx

    return net_dataidx_map


def iid_partition(all_targets, n_clients):
    labels = all_targets
    length = int(len(labels) / n_clients)
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    for client_idx in range(n_clients):
        np.random.shuffle(tot_idx)
        data_idxs = tot_idx[:length]
        tot_idx = tot_idx[length:]
        net_dataidx_map[client_idx] = np.array(data_idxs)

    return net_dataidx_map


def sharding_partition(all_targets, n_clients, shard_per_user, rand_set_all=[]):
    net_dataidx_map = {i: np.array([], dtype="int64") for i in range(n_clients)}
    idxs_dict = {}

    for i in range(len(all_targets)):
        label = torch.tensor(all_targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

        num_classes = len(np.unique(all_targets))
        shard_per_class = int(shard_per_user * n_clients / num_classes)

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((n_clients, -1))

    # divide and assign
    for i in range(n_clients):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        net_dataidx_map[i] = np.concatenate(rand_set).astype("int")

    test = []
    for key, value in net_dataidx_map.items():
        x = np.unique(torch.tensor(all_targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(all_targets)
    assert len(set(list(test))) == len(all_targets)

    return net_dataidx_map, rand_set_all


def sharding_max_partition(all_targets, n_clients, K):
    labels = all_targets
    length = int(len(labels) / n_clients)
    net_dataidx_map = {}

    shard_size = int(length / K)
    unique_classes = np.unique(labels)

    tot_idx_by_label = []
    for i in unique_classes:
        idx_by_label = np.where(labels == i)[0]
        tmp = []
        while 1:
            tmp.append(idx_by_label[:shard_size])
            idx_by_label = idx_by_label[shard_size:]
            if len(idx_by_label) < shard_size / 2:
                break
        tot_idx_by_label.append(tmp)

    for client_idx in range(n_clients):
        idx_by_devices = []

        while len(idx_by_devices) < K:
            chosen_label = np.random.choice(unique_classes, 1, replace=False)[
                0
            ]  # 임의의 Label을 하나 뽑음

            if (
                len(tot_idx_by_label[chosen_label]) > 0
            ):  # 만약 해당 Label의 shard가 하나라도 남아있다면,
                l_idx = np.random.choice(
                    len(tot_idx_by_label[chosen_label]), 1, replace=False
                )[
                    0
                ]  # shard 중 일부를 하나 뽑고
                idx_by_devices.append(
                    tot_idx_by_label[chosen_label][l_idx].tolist()
                )  # 클라이언트에 넣어준다.
                del tot_idx_by_label[chosen_label][l_idx]  # 뽑힌 shard의 원본은 제거!

        data_idxs = np.concatenate(idx_by_devices)
        np.random.shuffle(data_idxs)
        net_dataidx_map[client_idx] = data_idxs

    return net_dataidx_map


def lda_partition(all_targets, n_clients, alpha):
    labels = all_targets
    length = int(len(labels) / n_clients)
    net_dataidx_map = {}

    unique_classes = np.unique(labels)

    tot_idx_by_label = []
    for i in unique_classes:
        idx_by_label = np.where(labels == i)[0]
        tot_idx_by_label.append(idx_by_label)

    min_size = 0

    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        N, K = len(all_targets), len(np.unique(all_targets))

        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(all_targets == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(
                N, alpha, n_clients, idx_batch, idx_k
            )

    for i in range(n_clients):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def oracle_partition(all_targets, oracle_size=0):
    oracle_idxs = None

    if oracle_size != 0:
        idxs_dict = {}

        for i in range(len(all_targets)):
            label = torch.tensor(all_targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

            oracle_idxs = []

        for value in idxs_dict.values():

            oracle_idxs += value[0:oracle_size]

    return oracle_idxs


def get_dist_vec(dataloader, num_classes):
    """Calculate distribution vector for local set"""
    targets = dataloader.dataset.targets
    dist_vec = torch.zeros(num_classes)
    counter = Counter(targets)

    for class_idx, count in counter.items():
        dist_vec[class_idx] = count

    dist_vec /= len(targets)

    return dist_vec


def net_dataidx_map_counter(net_dataidx_map, all_targets):
    data_map = [[] for _ in range(len(net_dataidx_map.keys()))]
    num_classes = len(np.unique(all_targets))

    prev_key = -1
    for key, item in net_dataidx_map.items():
        client_class_count = [0 for _ in range(num_classes)]
        class_elems = all_targets[item]
        for elem in class_elems:
            client_class_count[elem] += 1

        data_map[key] = client_class_count

    return np.array(data_map)
