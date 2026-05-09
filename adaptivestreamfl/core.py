"""Command-line runner for the AdaptiveStreamFL reference implementation."""

from __future__ import annotations

import socket
from collections import deque
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .adaptive import (
    AdaptiveConfig,
    AdaptivePrototypeLearner,
    PrototypeAggregator,
    combine_stores,
    mean_or_default,
)
from .cli import apply_component_mode, args_parser
from .data import client_pro_streams, load_dataset, stream_client_data
from .microcluster import MicroClusterStore
from .runtime import DATASET_FEATURES, configure_standard_streams, on_off, set_random_seed


configure_standard_streams()


def result_file(output_dir: Path, *parts: str) -> Path:
    """Return an output path and create parent directories."""
    path = output_dir.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_labels(data: np.ndarray) -> np.ndarray:
    """Return a copy whose label column is encoded as contiguous integers."""
    normalized = np.asarray(data, dtype=float).copy()
    labels = normalized[:, -1]
    unique_labels = {label: idx for idx, label in enumerate(sorted(np.unique(labels)))}
    normalized[:, -1] = np.array([unique_labels[label] for label in labels], dtype=float)
    return normalized


def initialize_client_stores(
    proto_data: Dict[str, np.ndarray],
    feature_count: int,
    clusters_per_class: int,
    max_clusters: int,
    decay_rate: float,
    seed: int,
) -> Dict[str, MicroClusterStore]:
    """Create one local prototype store per client."""
    stores: Dict[str, MicroClusterStore] = {}
    for offset, (client_id, client_samples) in enumerate(proto_data.items()):
        store = MicroClusterStore(feature_count, max_clusters=max_clusters, decay_rate=decay_rate)
        store.initialize(client_samples, clusters_per_class=clusters_per_class, seed=seed + offset)
        stores[client_id] = store
    return stores


def recent_accuracy(history: Iterable[int], default: float = 1.0) -> float:
    values = list(history)
    if not values:
        return default
    return float(np.mean(values))


def run_stream(
    stream_data: Dict[str, np.ndarray],
    local_stores: Dict[str, MicroClusterStore],
    feature_count: int,
    output_dir: Path,
    dataset: str,
    client_count: int,
    max_clusters: int,
    global_max_clusters: int,
    reporting_interval: int,
    config: AdaptiveConfig,
    run_type: str,
) -> MicroClusterStore:
    """Run one federated stream pass and write step-level accuracy CSVs."""
    learner = AdaptivePrototypeLearner(config)
    aggregator = PrototypeAggregator(feature_count, max_clusters=global_max_clusters, config=config)
    global_store = MicroClusterStore(feature_count, max_clusters=global_max_clusters)
    hostname = socket.gethostname()

    client_ids = list(stream_data.keys())
    max_steps = min((len(stream_data[client_id]) for client_id in client_ids), default=0)
    correct_counts = {client_id: 0 for client_id in client_ids}
    recent_correct = {client_id: deque(maxlen=100) for client_id in client_ids}
    recent_uncertainty = {client_id: deque(maxlen=100) for client_id in client_ids}
    accuracy_rows: list[list[float]] = []

    for step_index in range(max_steps):
        client_scores: Dict[str, tuple[float, float]] = {}

        for client_id in client_ids:
            sample = stream_data[client_id][step_index]
            features = sample[:feature_count]
            label = int(sample[feature_count])
            local_store = local_stores[client_id]

            local_acc = recent_accuracy(recent_correct[client_id])
            local_uncertainty = mean_or_default(recent_uncertainty[client_id], default=0.0)

            if run_type == "AdaptiveStreamFL" and config.adaptive_mode:
                prediction_store = combine_stores(
                    feature_count,
                    local_store,
                    global_store if len(global_store) else None,
                    max_clusters=max_clusters + global_max_clusters,
                )
                k_value = learner.select_k(local_acc, local_uncertainty, len(prediction_store))
                radius_multiplier = learner.radius_multiplier(local_acc, local_uncertainty)
            else:
                prediction_store = local_store if not len(global_store) else combine_stores(
                    feature_count,
                    local_store,
                    global_store,
                    max_clusters=max_clusters + global_max_clusters,
                )
                k_value = 1
                radius_multiplier = 1.0

            prediction, uncertainty = prediction_store.predict(features, k=k_value)
            is_correct = int(prediction == label)
            correct_counts[client_id] += is_correct
            recent_correct[client_id].append(is_correct)
            recent_uncertainty[client_id].append(uncertainty)
            learner.remember_uncertainty(uncertainty)

            local_store.learn(features, label, timestamp=step_index, radius_multiplier=radius_multiplier)
            local_store.decay_reliability(current_time=step_index)

            cumulative_accuracy = correct_counts[client_id] / float(step_index + 1)
            client_scores[client_id] = (cumulative_accuracy, uncertainty)

        if (step_index + 1) % max(1, reporting_interval) == 0 or step_index + 1 == max_steps:
            global_store = aggregator.aggregate(local_stores, client_scores)
            row = [step_index + 1]
            for client_id in client_ids:
                row.append(round(correct_counts[client_id] / float(step_index + 1) * 100.0, 6))
            accuracy_rows.append(row)

    columns = ["step", *client_ids]
    results = pd.DataFrame(accuracy_rows, columns=columns)
    filename = f"{hostname}_{dataset}_{client_count}_{max_clusters}_{run_type.lower()}.csv"
    results.to_csv(result_file(output_dir, "fed", filename), index=False)

    if config.enable_component_tracking and accuracy_rows:
        mean_accuracy = float(np.mean(accuracy_rows[-1][1:]))
        config.record_component_metric(f"{config.component_mode}_performance", mean_accuracy)

    return global_store


def run(run_type: str, args) -> MicroClusterStore:
    """Load data, initialize clients, and run the requested mode."""
    feature_count = DATASET_FEATURES.get(args.dataset, args.features)
    data = normalize_labels(load_dataset(args.data_dir, args.dataset, feature_count))
    clients_data, _ = stream_client_data(data, num_clients=args.clients)
    stream_data, proto_data = client_pro_streams(clients_data, init_percent=args.percent_init)

    config = AdaptiveConfig()
    config.adaptive_mode = bool(args.adaptive_mode and run_type == "AdaptiveStreamFL")
    config.dynamic_k_adjustment = bool(args.dynamic_k_adjustment)
    config.smart_cluster_merge = bool(args.smart_cluster_merge)
    config.adaptive_weight_balance = bool(args.adaptive_weight_balance)
    config.performance_monitoring = bool(args.performance_monitoring)
    config.min_accuracy_threshold = float(args.min_accuracy_threshold)
    config.adaptation_frequency = int(args.adaptation_frequency)
    config.enable_prototype_learning = bool(args.enable_prototype_learning)
    config.enable_component_tracking = bool(args.enable_component_tracking)
    config.component_mode = args.component_mode
    config.prototype_quality_threshold = float(args.prototype_quality_threshold)

    if run_type != "AdaptiveStreamFL":
        apply_component_mode(config, "baseline")
        config.adaptive_mode = False
    elif config.enable_component_tracking:
        apply_component_mode(config, config.component_mode)

    local_stores = initialize_client_stores(
        proto_data=proto_data,
        feature_count=feature_count,
        clusters_per_class=args.local_init,
        max_clusters=args.max_mc,
        decay_rate=args.decay_rate,
        seed=args.seed,
    )

    print("AdaptiveStreamFL reference runner")
    print(f"dataset={args.dataset}, clients={args.clients}, mode={run_type}")
    print(f"adaptive_mode={on_off(config.adaptive_mode)}, dynamic_k={on_off(config.dynamic_k_adjustment)}")
    print(f"prototype_learning={on_off(config.enable_prototype_learning)}, uncertainty={on_off(config.enable_uncertainty_quantification)}")

    return run_stream(
        stream_data=stream_data,
        local_stores=local_stores,
        feature_count=feature_count,
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
        client_count=args.clients,
        max_clusters=args.max_mc,
        global_max_clusters=args.global_mc,
        reporting_interval=args.reporting_interval,
        config=config,
        run_type=run_type,
    )


def main(argv=None) -> None:
    args = args_parser(argv)
    set_random_seed(args.seed)

    run(args.run_type, args)


if __name__ == "__main__":
    main()
