# Compresso Demo

Standalone Streamlit demo workspace for exploring sparse representations and discovered clusters.

This repository is intentionally self-contained for Streamlit Cloud. It does
not install the private/full `compresso` library. Instead, it includes a tiny
read-only `compresso_demo_runtime` package that can load the exported SRP
tensor and cluster graph artifacts used by this demo.

Current data bundle:

```text
data/goodbooks/goodbooks_demo.zip
```

The bundle keeps only metadata, split information, sparse item representations,
and cluster graphs needed by the demo.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Validate from inside this directory:

```bash
python scripts/validate_bundle.py
```

Initial graph stages:

- `semantic_clustering_merged` - default explorer graph
- `topm_clustering_merged` - optional comparison graph
