from __future__ import annotations

import streamlit as st


def code(src: str, language: str = "python") -> None:
    st.code(src.strip(), language=language)


def render() -> None:
    st.title("Clustering")
    st.caption("Turning sparse representations into cluster graphs.")

    st.markdown(
        """
Compresso clustering is intentionally graph-oriented. A sparse representation
can create many meaningful overlapping groups: a book can belong to a Harry
Potter cluster, a fantasy cluster, a children's literature cluster, and a
movie-adaptation cluster at the same time.

The important distinction:

- **Clustering** creates initial cluster nodes from an `SRPTensor`.
- **Linking** connects existing nodes with parent/child relationships.
- **Merging** creates new parent nodes that combine several clusters.
- **Centroid merging** groups clusters by similarity in the same sparse feature space.
- **Coverage expansion** assigns otherwise uncovered entities to nearby clusters.
- **Pruning/filtering** changes which nodes are active or visible, without necessarily deleting the whole graph history.
- **Annotation** adds tags, labels, or descriptions.
"""
    )

    st.subheader("Core Data Model")
    code(
        """
@dataclass(frozen=True)
class SparseVector:
    indices: np.ndarray
    values: np.ndarray
    size: int

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
"""
    )
    st.markdown(
        """
**Meaning**

- `SparseVector`: sparse centroid or feature handle in the original SRP feature space.
- `SparseCluster`: one graph node. It has member entity row indices and a centroid.
- `SparseClusterSet`: whole cluster graph.
- `active_cluster_ids`: the current user-facing subset. The graph may contain many more nodes.
- `parent_cluster_ids` / `child_cluster_ids`: graph edges.

**Important `SparseClusterSet` properties / methods**

- `cluster_by_id`: dictionary from id to cluster.
- `active_clusters`: clusters listed in `active_cluster_ids`.
- `root_clusters`: clusters without parents.
- `leaf_clusters`: clusters without children.
- `entity_to_cluster_ids`: maps entity row to active clusters containing it.
- `entity_to_all_cluster_ids`: maps entity row to all graph nodes containing it.
- `children(cluster_id)`, `parents(cluster_id)`, `descendants(cluster_id)`, `ancestors(cluster_id)`.
- `with_active_cluster_ids(...)`: returns a graph with a different active set.
- `fill_missing_cluster_labels(...)`: optional utility to count/fill missing labels below roots.
"""
    )

    st.subheader("ClusteringPipeline")
    code(
        """
class ClusteringPipeline:
    def __init__(
        self,
        steps: Sequence[ClusterBuildStep | ClusterTransformStep],
        verbose: bool = False,
    ) -> None

    def fit(self, srp: SRPTensor) -> SparseClusterSet
    def __call__(self, srp: SRPTensor) -> SparseClusterSet
"""
    )
    st.markdown(
        """
**Parameters**

- `steps`: ordered pipeline. The first step consumes `SRPTensor` and creates
  `SparseClusterSet`; every following step consumes and returns `SparseClusterSet`.
- `verbose`: prints step-by-step progress summaries.

**Output**

Returns a `SparseClusterSet`.
"""
    )
    code(
        """
from compresso import clustering as cc

graph = cc.ClusteringPipeline(
    [
        cc.TopMSignedClustering(top_m=4, min_cluster_size=5),
        cc.EntityContainmentLink(threshold=1.0),
        cc.MaterializeLinkMerges(parent_scope="active"),
        cc.PruneRedundantRoots(),
        cc.SizeFilter(min_cluster_size=20),
    ],
    verbose=True,
).fit(srp)

print(len(graph.clusters), len(graph.active_clusters))
# Example output: 3003 378
"""
    )

    st.subheader("Cluster Builders")
    st.markdown("Builders are the only pipeline steps that consume an `SRPTensor` directly.")

    code(
        """
DominantSignedClustering(
    min_cluster_size: int = 1,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Creates one cluster per dominant signed feature. For each entity, find the
largest absolute activation and assign the entity to `feature:{idx}:{pos|neg}`.

- `min_cluster_size`: discard initial clusters smaller than this.
- `show_progress`: show tqdm progress bars.

Output: `SparseClusterSet` where clusters correspond to one signed feature.
"""
    )

    code(
        """
TopMSignedClustering(
    top_m: int = 1,
    min_cluster_size: int = 1,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Creates one cluster for each signed feature appearing in an entity's top `m`
activations. With `top_m=4`, an entity can be assigned to up to four feature
clusters.

- `top_m`: number of strongest signed features to consider per entity.
- `min_cluster_size`: discard clusters smaller than this.
- `show_progress`: show progress bars.

Output: overlapping feature clusters. This is broader than dominant clustering.
"""
    )

    code(
        """
ComboSignedClustering(
    top_m: int = 1,
    combo_size: int = 1,
    min_cluster_size: int = 1,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Builds clusters from combinations of signed features. Example: with
`top_m=3, combo_size=2`, entity features `[10+, 22-, 91+]` produce pair-combo
clusters such as `(10+,22-)`, `(10+,91+)`, `(22-,91+)`.

- `top_m`: features considered per entity.
- `combo_size`: exact size of feature combination.
- `min_cluster_size`: discard smaller combo clusters.
- `show_progress`: show progress bars.

Output: more specific intersection-like clusters than plain Top-M.
"""
    )

    code(
        """
FeaturePathClustering(
    top_m: int = 1,
    max_depth: int | None = None,
    min_cluster_size: int | None = None,
    min_activation: float | None = None,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Builds a hierarchy by recursively splitting entities by feature paths. A root
may represent one feature, its child a second feature among that subset, and so on.

- `top_m`: candidate features considered at each split.
- `max_depth`: maximum path depth. `None` means no explicit depth limit.
- `min_cluster_size`: stop or discard branches below this size. `None` disables.
- `min_activation`: minimum absolute activation required. Useful for normalized SRP.
- `show_progress`: show progress bars.

Output: graph with a natural feature-path hierarchy.
"""
    )

    code(
        """
SRPSimilarityClustering(
    threshold: float,
    top_k: int | None = 100,
    min_cluster_size: int = 2,
    normalize_rows: bool = True,
    min_local_density: float | None = None,
    centroid_top_k: int | None = None,
    batch_size: int = 1024,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Builds clusters from SRP-space cosine/dot-product neighborhoods. It creates
edges between similar entities, then connected components become clusters.

- `threshold`: minimum similarity for an edge.
- `top_k`: only consider top-k nearest neighbors per entity. `None` considers all pairs above threshold.
- `min_cluster_size`: discard connected components smaller than this.
- `normalize_rows`: normalize rows before similarity. When true, similarity is cosine-like.
- `min_local_density`: optional cleanup; remove entities whose within-component degree fraction is too low.
- `centroid_top_k`: keep only this many strongest centroid features. `None` keeps all non-zero centroid entries.
- `batch_size`: similarity computation batch size.
- `show_progress`: show progress bars.

Output: semantic/neighborhood clusters, often good for human-facing exploration.
"""
    )

    st.subheader("Linking")
    st.markdown(
        """
Linking adds parent/child edges between existing clusters. It does **not**
create a new combined cluster. This is useful when a small cluster is a subset
or specialization of a larger one, and you want hierarchy without losing either node.
"""
    )
    code(
        """
EntityContainmentLink(
    threshold: float = 1.0,
    child_scope: Literal["active", "all", "leaves", "roots"] = "leaves",
    parent_scope: Literal["active", "all", "leaves", "roots"] = "all",
    require_parent_larger: bool = True,
    skip_existing_ancestors: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Links child to parent when the child's entities are contained in the parent's
entities above `threshold`.

- `threshold`: containment score. `1.0` means every child entity must be in the parent.
- `child_scope`: which clusters can become children.
- `parent_scope`: which clusters can become parents.
- `require_parent_larger`: prevents same-size/smaller clusters from becoming parents.
- `skip_existing_ancestors`: avoids redundant links already implied by graph paths.
- `verbose`, `show_progress`: diagnostics.

Output: same nodes, additional edges.
"""
    )

    code(
        """
FeatureContainmentLink(
    threshold: float = 1.0,
    signed: bool = True,
    child_scope: Literal["active", "all", "leaves", "roots"] = "leaves",
    parent_scope: Literal["active", "all", "leaves", "roots"] = "all",
    require_parent_larger: bool = True,
    skip_existing_ancestors: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Links child to parent when the child's centroid feature set is contained in the
parent's feature set above `threshold`.

- `signed`: if true, feature sign matters (`feature:10:pos` differs from `feature:10:neg`).
- Other parameters mirror `EntityContainmentLink`.

Output: same nodes, additional edges based on feature support.
"""
    )

    st.subheader("Merging")
    st.markdown(
        """
Merging creates new cluster nodes that combine existing clusters. In Compresso,
this is generally **non-destructive**: original clusters remain as children or
source nodes unless a later compaction step removes/hides them for rendering.
"""
    )

    code(
        """
MaterializeLinkMerges(
    parent_scope: Literal["active", "all", "leaves", "roots"] = "active",
    include_descendants: bool = False,
    min_children: int = 1,
    normalize_centroids: bool = True,
    activate: bool = True,
    verbose: bool = False,
)
"""
    )
    st.markdown(
        """
Turns existing links into explicit parent merge nodes. This is useful after
linking, when you want combined cluster centroids for recommendation/steering.

- `parent_scope`: which linked parents should be materialized.
- `include_descendants`: include full descendant subtree, not only direct children.
- `min_children`: minimum number of children required to create a materialized node.
- `normalize_centroids`: normalize summed child centroid.
- `activate`: include new materialized nodes in `active_clusters`.
- `verbose`: print summary.

Output: graph with additional `merge:materialize_link_merges:*` nodes.
"""
    )

    code(
        """
EntityIoUMerge(
    threshold: float,
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Merges active clusters when their entity sets overlap by IoU above `threshold`.

- `threshold`: intersection-over-union threshold.
- `max_rounds`: repeat merging until stable or this many rounds.
- `normalize_centroids`: normalize merged centroid.
- `verbose`, `show_progress`: diagnostics.

Output: additional merged nodes; active set changes to merged representatives.
"""
    )

    code(
        """
EntityContainmentMerge(
    threshold: float = 1.0,
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown("Merges clusters when one entity set is contained in another. Parameters mirror `EntityIoUMerge`, but the score is containment rather than IoU.")

    code(
        """
FeatureContainmentMerge(
    threshold: float = 1.0,
    signed: bool = True,
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown("Merges clusters when centroid feature support is contained. `signed=True` treats positive and negative use of a feature as different.")

    code(
        """
CentroidSimilarityMerge(
    threshold: float,
    metric: Literal["cosine", "dot"] = "cosine",
    top_k: int | None = None,
    max_rounds: int = 10,
    min_group_size: int = 2,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Merges active clusters when their sparse centroids are similar. This is the
cluster-level analogue of `SRPSimilarityClustering`: instead of comparing
entity rows, it compares cluster centroid vectors.

- `threshold`: minimum centroid similarity.
- `metric`: `cosine` compares direction; `dot` compares raw centroid dot product.
- `top_k`: optional nearest-neighbor cap. `None` compares all centroid pairs above threshold.
- `max_rounds`: repeat merging until stable or this many rounds.
- `min_group_size`: minimum connected component size required for a merge.
- `normalize_centroids`: normalize merged parent centroid.
- `verbose`, `show_progress`: diagnostics.

Output: additional `merge:merge_clusters_by_centroid_similarity:*` parent
nodes, while original clusters remain as children.
"""
    )

    code(
        """
AssignUnclusteredToNearestCluster(
    srp: SRPTensor,
    metric: Literal["cosine", "dot"] = "cosine",
    min_similarity: float | None = None,
    top_k_clusters: int = 1,
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active",
    coverage_scope: Literal["active", "all"] = "active",
    assigned_weight: float = 1.0,
    centroid_top_k: int | None = None,
    normalize_centroids: bool = True,
    verbose: bool = False,
)
"""
    )
    st.markdown(
        """
Expands cluster coverage by assigning entities that are not covered by the
selected coverage scope to their nearest cluster centroids. This is a
**coverage step**, not a discovery step: it creates expanded parent nodes while
preserving original high-confidence clusters as children.

Graph shape:

```text
expanded_cluster
└── original_cluster
```

- `srp`: original entity SRP rows used for nearest-centroid assignment.
- `metric`: `cosine` or raw `dot` between entity row and cluster centroid.
- `min_similarity`: optional threshold. `None` means every uncovered entity is assigned.
- `top_k_clusters`: number of nearest clusters to assign each uncovered entity to.
- `cluster_scope`: which clusters can receive uncovered entities.
- `coverage_scope`: which clusters define whether an entity is already covered.
- `assigned_weight`: weight of newly assigned entities when recomputing the expanded centroid.
- `centroid_top_k`: optional truncation of expanded centroids.
- `normalize_centroids`: normalize expanded centroids.

Output: new expanded parent nodes with assignment provenance in `metadata`
and counts/similarity summaries in `stats`.
"""
    )

    code(
        """
TagSimilarityMerge(
    threshold: float,
    metric: Literal["weighted_jaccard", "cosine"] = "weighted_jaccard",
    max_rounds: int = 10,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown("Merges clusters with similar assigned tag vectors. Requires tags to be assigned first with `AssignTags`.")

    code(
        """
SemanticSimilarityMerge(
    embed_fn: Callable[[list[str]], np.ndarray],
    threshold: float = 0.9,
    text_fn: Callable[[SparseCluster], str] | None = None,
    label_fn: Callable[[object], object] | None = None,
    label_text_fn: Callable[[SparseCluster, list[SparseCluster]], object] | None = None,
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active",
    max_rounds: int = 10,
    min_group_size: int = 2,
    normalize_embeddings: bool = True,
    normalize_centroids: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
Creates parent clusters for semantically similar cluster labels/descriptions.
This is provider-agnostic: users supply `embed_fn`, and optionally `label_fn`
to name newly created semantic parent clusters.

- `embed_fn`: maps list of text strings to embedding matrix.
- `threshold`: similarity threshold.
- `text_fn`: converts cluster to text. Default uses description/label.
- `label_fn`: optional user function to name merged semantic parents.
- `label_text_fn`: optional function that builds text payload for naming a parent from its children.
- `cluster_scope`: which clusters are compared.
- `max_rounds`: repeat semantic grouping until stable.
- `min_group_size`: minimum number of similar clusters to create a parent.
- `normalize_embeddings`: normalize text embeddings before similarity.
- `normalize_centroids`: normalize parent centroid.

Output: added semantic parent nodes, e.g. `merge:semantic_similarity:*`.
"""
    )

    code(
        """
LabelDuplicateMerge(
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active",
    case_sensitive: bool = False,
    mark_children_hidden: bool = True,
    min_group_size: int = 2,
    normalize_centroids: bool = True,
    verbose: bool = False,
)

CompactHiddenClusters(
    hidden_key: str = "render_hidden",
    verbose: bool = False,
)
"""
    )
    st.markdown(
        """
`LabelDuplicateMerge` groups clusters with identical labels. It can mark
children as hidden instead of deleting them. `CompactHiddenClusters` then
physically removes hidden nodes for cleaner rendering.

Use this when semantic labeling creates many equivalent labels that should
appear as one cluster in a UI.
"""
    )

    st.subheader("Pruning and Filtering")
    code(
        """
PruneRedundantRoots(verbose: bool = False)

SizeFilter(min_cluster_size: int)
"""
    )
    st.markdown(
        """
- `PruneRedundantRoots`: removes active root clusters that are already represented
  by another active branch. It changes `active_cluster_ids`; it does not erase all graph history.
- `SizeFilter`: keeps only active clusters with at least `min_cluster_size` entities.

These steps are for presentation and usability: they decide what the explorer
or downstream user sees as active.
"""
    )

    st.subheader("Annotation")
    code(
        """
AssignTags(
    entity_tag_matrix: object,
    tag_names: Sequence[str],
    method: Literal["tfidf", "counts"] = "tfidf",
    top_k: int = 5,
    min_score: float = 0.0,
)

LabelClusters(
    entity_metadata: object,
    text_extractor: Callable[[SparseCluster, object], object],
    label_fn: Callable[[object], object],
    cluster_scope: Literal["active", "all", "leaves", "roots"] = "active",
    overwrite: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
)
"""
    )
    st.markdown(
        """
`AssignTags` assigns tag summaries from an entity-tag matrix. `method="tfidf"`
downweights globally common tags, while `method="counts"` uses raw counts.

`LabelClusters` is intentionally callback-based:

- `text_extractor(cluster, entity_metadata)`: builds a text payload from cluster members.
- `label_fn(payload)`: returns a label string, or a mapping with label/description.
- `cluster_scope`: which nodes are labeled.
- `overwrite`: if false, existing labels are preserved.

This keeps Compresso independent of LLM providers, API keys, sentence-transformer
choices, and domain-specific prompts.
"""
    )

    st.subheader("Convenience API")
    code(
        """
def cluster_srp(
    srp: SRPTensor,
    *,
    mode: Literal["dominant_signed", "top_m_signed", "combo_signed"] = "dominant_signed",
    top_m: int = 1,
    combo_size: int = 1,
    min_cluster_size: int = 1,
    post_merge_min_cluster_size: int | None = None,
    entity_ids: np.ndarray | None = None,
    activation_iou_threshold: float | None = None,
    entity_tag_matrix=None,
    tag_names: Sequence[str] | None = None,
    tag_method: Literal["tfidf", "counts"] = "tfidf",
    top_k_tags: int = 5,
    tag_similarity_threshold: float | None = None,
    tag_similarity_metric: Literal["weighted_jaccard", "cosine"] = "weighted_jaccard",
    max_merge_rounds: int = 10,
    verbose: bool = False,
    show_progress: bool = False,
) -> SparseClusterSet
"""
    )
    st.markdown(
        """
`cluster_srp` is a compact convenience wrapper for simple activation-based
clustering. For more explicit and reproducible work, prefer `ClusteringPipeline`.
"""
    )

    st.subheader("Saving and Loading")
    code(
        """
from compresso.clustering import save_cluster_graph, load_cluster_graph

save_cluster_graph(graph, "graph.json")
graph = load_cluster_graph("graph.json")
"""
    )
    st.markdown(
        """
Cluster graphs are saved as JSON with cluster nodes, active IDs, labels,
metadata, parent/child links, and sparse centroids. This is the format used by
the demo bundle.
"""
    )

    st.subheader("Typical Pipeline Shape")
    code(
        """
graph = cc.ClusteringPipeline(
    [
        # 1. Build initial nodes from sparse vectors.
        cc.SRPSimilarityClustering(
            threshold=0.45,
            top_k=100,
            min_cluster_size=5,
            normalize_rows=True,
            centroid_top_k=4,
        ),

        # 2. Add hierarchy between existing nodes.
        cc.EntityContainmentLink(threshold=1.0),
        cc.FeatureContainmentLink(threshold=1.0),

        # 3. Create explicit parent nodes from links.
        cc.MaterializeLinkMerges(parent_scope="active"),

        # 4. Optionally group nearby cluster centroids.
        cc.CentroidSimilarityMerge(threshold=0.85, top_k=20),

        # 5. Optionally expand final coverage.
        cc.AssignUnclusteredToNearestCluster(srp, coverage_scope="all"),

        # 6. Choose active visible clusters.
        cc.PruneRedundantRoots(),
        cc.SizeFilter(min_cluster_size=20),

        # 7. Add human-readable names.
        cc.LabelClusters(...),

        # 8. Optional semantic grouping.
        cc.SemanticSimilarityMerge(...),
    ]
).fit(srp)
"""
    )
