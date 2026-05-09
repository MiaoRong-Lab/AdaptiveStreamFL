"""Dataset loading and stream partition helpers."""

from __future__ import annotations

import collections
import random
from pathlib import Path

import numpy as np


def load_dataset(data_dir, dataset_name, expected_features=None):
    """Load a prepared .npy dataset whose last column is the label."""
    data_path = Path(data_dir) / f"{dataset_name}.npy"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}. "
            "Use --data_dir to point to prepared .npy files."
        )

    data = np.load(data_path)
    print(data.shape)

    if expected_features is not None and data.shape[1] - 1 != expected_features:
        print(
            f"[warning] Dataset feature count mismatch: "
            f"expected {expected_features}, got {data.shape[1] - 1}"
        )

    return np.asarray(data).astype(float)


def initial_class_data(data, init_percent, label_index):
    """Return initial samples grouped by class label."""
    class_data = {}
    limit_size = int(init_percent * len(data))
    initial_load = data[0:limit_size, :]
    all_classes = np.unique(initial_load[:, label_index])

    for class_label in list(all_classes):
        class_data[int(class_label)] = initial_load[initial_load[:, label_index] == class_label]

    return class_data


def partition_by_class(initial_load, label_index):
    """Group an existing initial sample array by class label."""
    class_data = {}
    all_classes = np.unique(initial_load[:, label_index])
    for class_label in list(all_classes):
        class_data[int(class_label)] = initial_load[initial_load[:, label_index] == class_label]
    return class_data


def stream_slice(data, init_percent):
    """Return the stream portion after the initial prototype slice."""
    initial_size = int(init_percent * len(data))
    stream_load = data[initial_size + 1 : len(data) + 1, :]
    print(stream_load.shape)
    return stream_load


def client_data_random(data, num_clients=10, client_initial="c"):
    """Randomly shard data into client partitions."""
    client_names = ["{}_{}".format(client_initial, i + 1) for i in range(num_clients)]
    random.shuffle(data)
    size = len(data) // num_clients
    shards = [data[i : i + size] for i in range(0, size * num_clients, size)]
    assert len(shards) == len(client_names)
    return {client_names[i]: shards[i] for i in range(len(client_names))}, client_names


def stream_client_data(data, num_clients=10, client_initial="c"):
    """Round-robin stream data into client partitions."""
    order_dict = collections.OrderedDict()
    client_names = ["{}_{}".format(client_initial, i + 1) for i in range(num_clients)]
    shards = [[] for _ in range(0, num_clients)]

    counter_check = 0
    for i in range(0, len(data), 1):
        shards[counter_check].append(data[i])
        counter_check = counter_check + 1

        if (i + 1) % num_clients == 0:
            counter_check = 0

    assert len(shards) == len(client_names)

    for i in range(len(client_names)):
        order_dict[client_names[i]] = np.asarray(shards[i])

    return order_dict, client_names


def client_pro_streams(clients_data, init_percent):
    """Split each client's partition into prototype and stream samples."""
    proto_data = collections.OrderedDict()
    stream_data = collections.OrderedDict()

    for client_key in clients_data.keys():
        client_records = len(clients_data[client_key])
        client_init = int(init_percent * client_records)
        proto_data[client_key] = clients_data[client_key][0:client_init, :]
        stream_data[client_key] = clients_data[client_key][client_init + 1 : client_records + 1, :]

    return stream_data, proto_data
