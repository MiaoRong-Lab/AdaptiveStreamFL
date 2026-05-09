"""Adaptive control components for AdaptiveStreamFL.

The classes in this module are small policy objects used by the streaming
runner. They implement adaptive neighbour selection, radius scaling, and
uncertainty-aware aggregation without depending on external project code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import numpy as np

from .microcluster import MicroCluster, MicroClusterStore


@dataclass
class AdaptiveConfig:
    """Configuration for AdaptiveStreamFL component switches."""

    adaptive_mode: bool = True
    dynamic_k_adjustment: bool = True
    enable_prototype_learning: bool = True
    enable_uncertainty_quantification: bool = True
    enable_hierarchical_federated_learning: bool = True
    enable_component_tracking: bool = False
    component_mode: str = "full"
    smart_cluster_merge: bool = True
    adaptive_weight_balance: bool = True
    performance_monitoring: bool = True
    min_accuracy_threshold: float = 0.85
    adaptation_frequency: int = 100
    k_values: Sequence[int] = (1, 3, 5, 7)
    current_best_k: int = 1
    prototype_quality_threshold: float = 0.8
    component_tracking: Dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_tracking:
            self.component_tracking = {
                "baseline_performance": [],
                "k_only_performance": [],
                "prototype_only_performance": [],
                "full_performance": [],
            }
        self.experiment_metadata = {
            "algorithm_name": "AdaptiveStreamFL",
            "version": "0.1.0",
            "method_family": "Bayesian adaptive prototype learning",
            "key_components": [
                "dynamic_k_selection",
                "multi_scale_prototype_memory",
                "uncertainty_weighted_aggregation",
            ],
        }

    def record_component_metric(self, name: str, value: float, limit: int = 200) -> None:
        self.component_tracking.setdefault(name, []).append(float(value))
        if len(self.component_tracking[name]) > limit:
            self.component_tracking[name] = self.component_tracking[name][-limit:]


class AdaptivePrototypeLearner:
    """Policy object for stream-level adaptive prototype decisions."""

    def __init__(self, config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()
        self.k_history: list[int] = []
        self.uncertainty_history: list[float] = []
        self.radius_history: list[float] = []

    def select_k(self, recent_accuracy: float, recent_uncertainty: float, cluster_count: int) -> int:
        """Choose the number of nearest prototypes used for prediction."""
        if not self.config.adaptive_mode or not self.config.dynamic_k_adjustment:
            return 1

        if cluster_count <= 1:
            chosen = 1
        elif recent_uncertainty > 0.45 or recent_accuracy < self.config.min_accuracy_threshold:
            chosen = max(self.config.k_values)
        elif recent_uncertainty > 0.25:
            chosen = sorted(self.config.k_values)[len(self.config.k_values) // 2]
        else:
            chosen = min(self.config.k_values)

        chosen = int(max(1, min(chosen, max(1, cluster_count))))
        self.config.current_best_k = chosen
        self.k_history.append(chosen)
        return chosen

    def radius_multiplier(self, recent_accuracy: float, recent_uncertainty: float) -> float:
        """Scale prototype update radii from recent stream feedback."""
        if not self.config.adaptive_mode or not self.config.enable_prototype_learning:
            return 1.0

        multiplier = 1.0
        if recent_accuracy < self.config.min_accuracy_threshold:
            multiplier += 0.25
        if recent_uncertainty > 0.4:
            multiplier += 0.25
        if recent_accuracy > 0.95 and recent_uncertainty < 0.2:
            multiplier -= 0.15

        multiplier = float(max(0.6, min(1.8, multiplier)))
        self.radius_history.append(multiplier)
        return multiplier

    def aggregation_weight(self, accuracy: float, uncertainty: float, cluster_count: int) -> float:
        """Compute a non-negative client weight for prototype aggregation."""
        reliability = max(0.0, float(accuracy))
        confidence = 1.0 - max(0.0, min(1.0, float(uncertainty)))
        memory = np.log1p(max(0, int(cluster_count)))
        return float(max(0.01, reliability * confidence * (1.0 + memory)))

    def remember_uncertainty(self, uncertainty: float) -> None:
        self.uncertainty_history.append(float(uncertainty))
        if len(self.uncertainty_history) > 500:
            self.uncertainty_history = self.uncertainty_history[-500:]


class PrototypeAggregator:
    """Build a global prototype store from client stores."""

    def __init__(self, feature_count: int, max_clusters: int, config: AdaptiveConfig | None = None):
        self.feature_count = int(feature_count)
        self.max_clusters = int(max_clusters)
        self.config = config or AdaptiveConfig()

    def aggregate(
        self,
        client_stores: Dict[str, MicroClusterStore],
        client_scores: Dict[str, tuple[float, float]],
    ) -> MicroClusterStore:
        """Return a global store containing weighted client prototypes."""
        global_store = MicroClusterStore(self.feature_count, max_clusters=self.max_clusters)
        learner = AdaptivePrototypeLearner(self.config)

        weighted_clusters: list[tuple[float, MicroCluster]] = []
        for client_id, store in client_stores.items():
            accuracy, uncertainty = client_scores.get(client_id, (0.0, 1.0))
            weight = learner.aggregation_weight(accuracy, uncertainty, len(store))
            for cluster in store.top_clusters(max(1, self.max_clusters // max(1, len(client_stores)))):
                copied = cluster.copy()
                copied.reliability *= weight
                weighted_clusters.append((copied.reliability, copied))

        for _, cluster in sorted(weighted_clusters, key=lambda item: item[0], reverse=True):
            global_store.add(cluster)

        return global_store


def combine_stores(
    feature_count: int,
    local_store: MicroClusterStore,
    global_store: MicroClusterStore | None,
    max_clusters: int,
) -> MicroClusterStore:
    """Create a temporary prediction store from local and global prototypes."""
    combined = MicroClusterStore(feature_count, max_clusters=max_clusters)
    combined.extend(local_store.clusters)
    if global_store is not None:
        combined.extend(global_store.clusters)
    return combined


def mean_or_default(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float(default)
