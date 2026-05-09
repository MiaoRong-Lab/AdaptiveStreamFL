"""Micro-cluster primitives for AdaptiveStreamFL.

This module implements a compact prototype store for streaming classification.
Each micro-cluster keeps sufficient statistics for one labelled region of the
feature space and can be updated incrementally as new samples arrive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.cluster import KMeans


EPSILON = 1e-12


@dataclass
class MicroCluster:
    """Incremental labelled prototype."""

    label: int
    linear_sum: np.ndarray
    squared_sum: np.ndarray
    weight: float
    last_update: int = 0
    reliability: float = 1.0

    @classmethod
    def from_points(cls, points: np.ndarray, label: int, timestamp: int = 0) -> "MicroCluster":
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return cls(
            label=int(label),
            linear_sum=points.sum(axis=0),
            squared_sum=np.square(points).sum(axis=0),
            weight=float(points.shape[0]),
            last_update=timestamp,
            reliability=1.0,
        )

    @property
    def centroid(self) -> np.ndarray:
        return self.linear_sum / max(self.weight, EPSILON)

    @property
    def radius(self) -> float:
        mean_square = self.squared_sum / max(self.weight, EPSILON)
        variance = np.maximum(mean_square - np.square(self.centroid), 0.0)
        return float(np.sqrt(np.mean(variance) + EPSILON))

    def distance(self, point: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - self.centroid))

    def update(self, point: Sequence[float], timestamp: int) -> None:
        point = np.asarray(point, dtype=float)
        self.linear_sum = self.linear_sum + point
        self.squared_sum = self.squared_sum + np.square(point)
        self.weight += 1.0
        self.last_update = timestamp
        self.reliability += 1.0

    def decay(self, current_time: int, decay_rate: float) -> None:
        age = max(0, current_time - self.last_update)
        self.reliability *= 2 ** (-decay_rate * age)

    def copy(self) -> "MicroCluster":
        return MicroCluster(
            label=self.label,
            linear_sum=self.linear_sum.copy(),
            squared_sum=self.squared_sum.copy(),
            weight=float(self.weight),
            last_update=int(self.last_update),
            reliability=float(self.reliability),
        )


class MicroClusterStore:
    """A mutable collection of labelled micro-clusters."""

    def __init__(self, feature_count: int, max_clusters: int = 200, decay_rate: float = 0.0):
        self.feature_count = int(feature_count)
        self.max_clusters = int(max_clusters)
        self.decay_rate = float(decay_rate)
        self.clusters: List[MicroCluster] = []

    def __len__(self) -> int:
        return len(self.clusters)

    def add(self, cluster: MicroCluster) -> None:
        self.clusters.append(cluster.copy())
        self.prune_or_merge()

    def extend(self, clusters: Iterable[MicroCluster]) -> None:
        for cluster in clusters:
            self.clusters.append(cluster.copy())
        self.prune_or_merge()

    def initialize(self, labelled_data: np.ndarray, clusters_per_class: int, seed: int = 0) -> None:
        labelled_data = np.asarray(labelled_data, dtype=float)
        if labelled_data.size == 0:
            return

        labels = np.unique(labelled_data[:, -1]).astype(int)
        for label in labels:
            class_points = labelled_data[labelled_data[:, -1].astype(int) == label, :-1]
            if class_points.size == 0:
                continue
            cluster_count = max(1, min(int(clusters_per_class), class_points.shape[0]))
            if cluster_count == 1:
                self.add(MicroCluster.from_points(class_points, int(label)))
                continue

            kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
            assignments = kmeans.fit_predict(class_points)
            for cluster_id in range(cluster_count):
                members = class_points[assignments == cluster_id]
                if members.size:
                    self.add(MicroCluster.from_points(members, int(label)))

    def nearest(self, point: Sequence[float], label: int | None = None) -> tuple[int | None, MicroCluster | None, float]:
        candidates = [
            (idx, cluster, cluster.distance(point))
            for idx, cluster in enumerate(self.clusters)
            if label is None or cluster.label == int(label)
        ]
        if not candidates:
            return None, None, float("inf")
        return min(candidates, key=lambda item: item[2])

    def predict(self, point: Sequence[float], k: int = 1) -> tuple[int, float]:
        if not self.clusters:
            return 0, 1.0

        distances = np.array([cluster.distance(point) for cluster in self.clusters], dtype=float)
        order = np.argsort(distances)[: max(1, min(k, len(self.clusters)))]

        votes: dict[int, float] = {}
        for idx in order:
            cluster = self.clusters[int(idx)]
            scale = max(cluster.radius, EPSILON)
            vote = cluster.reliability / (distances[int(idx)] / scale + 1.0)
            votes[cluster.label] = votes.get(cluster.label, 0.0) + float(vote)

        predicted = max(votes.items(), key=lambda item: item[1])[0]
        selected_distances = distances[order]
        uncertainty = float(np.min(selected_distances) / (np.mean(selected_distances) + 1.0))
        return int(predicted), max(0.0, min(1.0, uncertainty))

    def learn(self, point: Sequence[float], label: int, timestamp: int, radius_multiplier: float = 1.0) -> None:
        idx, cluster, distance = self.nearest(point, label=int(label))
        if cluster is None:
            self.add(MicroCluster.from_points(np.asarray(point, dtype=float), int(label), timestamp))
            return

        acceptance_radius = max(cluster.radius * radius_multiplier, 1e-6)
        if distance <= acceptance_radius or cluster.weight < 2:
            cluster.update(point, timestamp)
        else:
            self.add(MicroCluster.from_points(np.asarray(point, dtype=float), int(label), timestamp))
        self.prune_or_merge()

    def decay_reliability(self, current_time: int, threshold: float = 0.01) -> None:
        if self.decay_rate <= 0:
            return
        for cluster in self.clusters:
            cluster.decay(current_time, self.decay_rate)
        self.clusters = [cluster for cluster in self.clusters if cluster.reliability >= threshold]

    def prune_or_merge(self) -> None:
        while len(self.clusters) > self.max_clusters:
            self._merge_closest_pair()

    def _merge_closest_pair(self) -> None:
        if len(self.clusters) <= 1:
            return

        best_pair = None
        best_distance = float("inf")
        for i, first in enumerate(self.clusters):
            for j in range(i + 1, len(self.clusters)):
                second = self.clusters[j]
                if first.label != second.label:
                    continue
                distance = first.distance(second.centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        if best_pair is None:
            self.clusters.sort(key=lambda cluster: cluster.reliability)
            self.clusters.pop(0)
            return

        i, j = best_pair
        first = self.clusters[i]
        second = self.clusters[j]
        merged = MicroCluster(
            label=first.label,
            linear_sum=first.linear_sum + second.linear_sum,
            squared_sum=first.squared_sum + second.squared_sum,
            weight=first.weight + second.weight,
            last_update=max(first.last_update, second.last_update),
            reliability=max(first.reliability, second.reliability),
        )
        for index in sorted(best_pair, reverse=True):
            self.clusters.pop(index)
        self.clusters.append(merged)

    def top_clusters(self, limit: int) -> list[MicroCluster]:
        ordered = sorted(self.clusters, key=lambda cluster: (cluster.reliability, cluster.weight), reverse=True)
        return [cluster.copy() for cluster in ordered[: max(0, int(limit))]]
