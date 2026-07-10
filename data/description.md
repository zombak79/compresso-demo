# Methodology

## Dataset

This example uses Amazon Reviews 2023 dataset. For all categories, see this [link](https://zombak79.github.io/compresso-recsys/cli-reference.html#supported-amazon-reviews-2023-datasets).  We build a compact recommender-system checkpoint with item metadata, interaction splits, and product image URLs.

```python
import compresso
import compresso_recsys as cr
import pandas as pd

from compresso import clustering as cc
from compresso import TopKSAETrainer, TopKSAEConfig, L1Normalize

category = "..."

checkpoint_path = cr.build_recsys_checkpoint(
    dataset="amazon2023",
    amazon_category=category,
    checkpoint_path=f"artifacts/{category.lower()}.zip",
    metadata_text_fields=["title", "features", "description", "categories"],
    split_mode="item_split",
    min_entity_text_words=20,
    val_items=500,
    test_items=1000,
    min_user_support=10,
    item_min_support=10,
    min_value_to_keep=1.0,
    set_all_values_to=1.0,
    min_source_items=1,
    min_target_items=1,
    annotation_source="none",
    show_progress=True,
    include_image_urls=True,
)
```

We then load the item metadata from the checkpoint.

```python
with cr.read_checkpoint(checkpoint_path) as root:
    split = cr.load_recsys_split(root)
    meta = split["entity_metadata"]
```

## Sparse Representation

Product metadata is encoded with `sentence-transformers/all-MiniLM-L6-v2`, producing dense semantic item embeddings.

```python
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

item_embeddings = sbert.encode(
    meta.entity_text,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

We then train a Top-k sparse autoencoder with Compresso and export the sparse item representations.

```python
sae = TopKSAETrainer(
    TopKSAEConfig(
        hidden_dim=2048,
        k=32,
        batch_size=1024,
        epochs=100,
        lr=1e-3,
        decay=True,
        post_sparsify=L1Normalize(),
        sparsify_score_mode="abs",
        sparsify_ste_alpha=0.01,
        device="cuda",
    )
)

sparse_embeddings = sae.fit_transform(item_embeddings)
```

## Clustering and Labeling

We use `DominantSignedClustering`, which treats sparse representations as an inverted index. Items whose strongest sparse activation is the same signed latent factor are grouped into one cluster.

```python
cluster_graph = cc.DominantSignedClustering(
    min_cluster_size=150,
    show_progress=True,
)(sparse_embeddings)

print(
    f"Number of discovered clusters: {len(cluster_graph.clusters)}, "
    f"active clusters: {len(cluster_graph.active_clusters)}, "
    f"root clusters: {len(cluster_graph.root_clusters)}"
)
```

```text
Number of discovered clusters: 60, active clusters: 60, root clusters: 60
```

To label each cluster, we first extract representative item titles from its members.

```python
def item_texts(
    cluster: cc.types.SparseCluster,
    metadata: pd.DataFrame,
    limit: int = 50,
) -> str:
    rows = metadata.iloc[cluster.entity_indices]

    if len(rows) > limit:
        rows = rows.sample(limit)

    return "\n".join(rows["title"].fillna("").tolist())
```

Then we ask a local LLM to convert those titles into a concise recommender-system segment name. This example uses `ollama`, but the callback can be adapted to any LLM provider.

```python
import ollama

MODEL = "gpt-oss:20b"

def label_cluster(items: str) -> str:
    prompt = f"""You are given a list of items, each with metadata.

1. Identify the most common concept that links all the items.
2. Convert that concept into a segment name suitable for a recommendation system.

The segment name must be title-case, concise (2-5 words), and sound natural,
like a product category. Prefer distinctive labels over generic ones.

Output only the final segment name.

Items:
{items}

Segment name:"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict classification model. "
                    "Return only one segment name. Never explain."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 1000,
            "num_ctx": 8192,
        },
        think="low",
        stream=False,
    )

    return response["message"]["content"] or ""
```

Finally, we apply the labeling step to all clusters.

```python
cluster_graph = cc.LabelClusters(
    entity_metadata=meta,
    text_extractor=item_texts,
    label_fn=label_cluster,
    cluster_scope="all",
    show_progress=True,
)(cluster_graph)
```

The clustering and labeling steps can also be composed into one pipeline.

```python
cluster_graph = cc.ClusteringPipeline(
    [
        cc.DominantSignedClustering(
            min_cluster_size=150,
            show_progress=True,
        ),
        cc.LabelClusters(
            entity_metadata=meta,
            text_extractor=item_texts,
            label_fn=label_cluster,
            cluster_scope="all",
            show_progress=True,
        ),
    ]
)(sparse_embeddings)
```

## Export

The demo only needs a compact table: the cluster label, the dominant latent factor, and the item row indices that belong to the cluster.

```python
data = [
    (
        str(cluster.label or "").strip(),
        [cluster.centroid.indices.item(), cluster.centroid.values.item()],
        cluster.entity_indices,
    )
    for cluster in cluster_graph.clusters
]

df = pd.DataFrame(data, columns=["cluster_label", "centroid", "indices"])
```
