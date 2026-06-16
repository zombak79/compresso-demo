from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np


FORMAT = "compresso.sparse_cluster_graph"
VERSION = 1


@dataclass(frozen=True)
class SparseVector:
    indices: np.ndarray
    values: np.ndarray
    size: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.float32)
        if indices.ndim != 1 or values.ndim != 1:
            raise ValueError("SparseVector indices and values must be 1D")
        if indices.shape[0] != values.shape[0]:
            raise ValueError("SparseVector indices and values must have the same length")
        if int(self.size) <= 0:
            raise ValueError("SparseVector size must be positive")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "size", int(self.size))


@dataclass(frozen=True)
class ScoredTag:
    tag_id: int
    name: str
    score: float
    count: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SparseCluster:
    cluster_id: str
    centroid: SparseVector
    entity_indices: np.ndarray
    source_cluster_ids: tuple[str, ...] = ()
    parent_cluster_ids: tuple[str, ...] = ()
    child_cluster_ids: tuple[str, ...] = ()
    tags: tuple[ScoredTag, ...] = ()
    label: str | None = None
    description: str | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entities = np.asarray(self.entity_indices, dtype=np.int64)
        if entities.ndim != 1:
            raise ValueError("SparseCluster.entity_indices must be 1D")
        object.__setattr__(self, "entity_indices", np.unique(entities))

    @property
    def entity_count(self) -> int:
        return int(self.entity_indices.size)


@dataclass(frozen=True)
class SparseClusterSet:
    clusters: tuple[SparseCluster, ...]
    n_entities: int
    n_features: int
    active_cluster_ids: tuple[str, ...] | None = None
    entity_ids: np.ndarray | None = None
    feature_ids: np.ndarray | None = None
    assignment_mode: str = "dominant_signed"
    history: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_entities", int(self.n_entities))
        object.__setattr__(self, "n_features", int(self.n_features))
        ids = tuple(cluster.cluster_id for cluster in self.clusters)
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_id values must be unique")
        active = ids if self.active_cluster_ids is None else tuple(str(v) for v in self.active_cluster_ids)
        missing = [cluster_id for cluster_id in active if cluster_id not in set(ids)]
        if missing:
            raise ValueError(f"active_cluster_ids contains unknown ids: {missing[:5]}")
        object.__setattr__(self, "active_cluster_ids", active)

    @property
    def cluster_by_id(self) -> dict[str, SparseCluster]:
        return {cluster.cluster_id: cluster for cluster in self.clusters}

    @property
    def active_clusters(self) -> tuple[SparseCluster, ...]:
        by_id = self.cluster_by_id
        return tuple(by_id[cluster_id] for cluster_id in (self.active_cluster_ids or ()))

    @property
    def root_clusters(self) -> tuple[SparseCluster, ...]:
        return tuple(cluster for cluster in self.clusters if not cluster.parent_cluster_ids)

    @property
    def entity_to_cluster_ids(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for cluster in self.active_clusters:
            for entity_idx in cluster.entity_indices.tolist():
                out.setdefault(int(entity_idx), []).append(cluster.cluster_id)
        return out

    @property
    def entity_to_all_cluster_ids(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for cluster in self.clusters:
            for entity_idx in cluster.entity_indices.tolist():
                out.setdefault(int(entity_idx), []).append(cluster.cluster_id)
        return out

    def children(self, cluster_id: str) -> tuple[SparseCluster, ...]:
        by_id = self.cluster_by_id
        return tuple(by_id[child_id] for child_id in by_id[cluster_id].child_cluster_ids)

    def parents(self, cluster_id: str) -> tuple[SparseCluster, ...]:
        by_id = self.cluster_by_id
        return tuple(by_id[parent_id] for parent_id in by_id[cluster_id].parent_cluster_ids)


def _tag_from_dict(data: Mapping[str, Any]) -> ScoredTag:
    return ScoredTag(
        tag_id=int(data["tag_id"]),
        name=str(data["name"]),
        score=float(data["score"]),
        count=float(data.get("count", 0.0)),
        metadata=dict(data.get("metadata", {})),
    )


def _cluster_from_dict(data: Mapping[str, Any]) -> SparseCluster:
    centroid = data["centroid"]
    return SparseCluster(
        cluster_id=str(data["cluster_id"]),
        centroid=SparseVector(
            np.asarray(centroid["indices"], dtype=np.int64),
            np.asarray(centroid["values"], dtype=np.float32),
            int(centroid["size"]),
        ),
        entity_indices=np.asarray(data["entity_indices"], dtype=np.int64),
        source_cluster_ids=tuple(str(v) for v in data.get("source_cluster_ids", ())),
        parent_cluster_ids=tuple(str(v) for v in data.get("parent_cluster_ids", ())),
        child_cluster_ids=tuple(str(v) for v in data.get("child_cluster_ids", ())),
        tags=tuple(_tag_from_dict(tag) for tag in data.get("tags", ())),
        label=data.get("label"),
        description=data.get("description"),
        stats=dict(data.get("stats", {})),
        metadata=dict(data.get("metadata", {})),
    )


def graph_from_dict(data: Mapping[str, Any]) -> SparseClusterSet:
    if data.get("format") != FORMAT:
        raise ValueError(f"Unsupported cluster graph format: {data.get('format')!r}")
    if int(data.get("version", -1)) != VERSION:
        raise ValueError(f"Unsupported cluster graph version: {data.get('version')!r}")
    return SparseClusterSet(
        clusters=tuple(_cluster_from_dict(cluster) for cluster in data["clusters"]),
        n_entities=int(data["n_entities"]),
        n_features=int(data["n_features"]),
        active_cluster_ids=tuple(str(v) for v in data.get("active_cluster_ids", ())),
        entity_ids=np.asarray(data["entity_ids"]) if data.get("entity_ids") is not None else None,
        feature_ids=np.asarray(data["feature_ids"]) if data.get("feature_ids") is not None else None,
        assignment_mode=str(data.get("assignment_mode", "dominant_signed")),
        history=tuple(dict(entry) for entry in data.get("history", ())),
        metadata=dict(data.get("metadata", {})),
    )


def load_cluster_graph(path: str | Path) -> SparseClusterSet:
    return graph_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
