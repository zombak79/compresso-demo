from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import json
import re
import zipfile

import pandas as pd

from .paths import DATA_DIR


AMAZON_CLUSTER_FORMAT = "compresso.demo.amazon_clusters"


@dataclass(frozen=True)
class ExampleBundleInfo:
    path: Path
    dataset_id: str
    title: str
    format: str
    version: int

    @property
    def page_key(self) -> str:
        return f"example:{self.dataset_id}"


@dataclass(frozen=True)
class DescriptionSection:
    title: str
    anchor: str


def _slugify_heading(text: str) -> str:
    slug = re.sub(r"[^a-z0-9 -]", "", text.strip().lower())
    return re.sub(r"\s+", "-", slug).strip("-")


def markdown_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or None
    return None


def markdown_sections(markdown: str) -> list[DescriptionSection]:
    sections: list[DescriptionSection] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            if title:
                sections.append(DescriptionSection(title=title, anchor=_slugify_heading(title)))
    return sections


def markdown_without_title(markdown: str) -> str:
    lines = markdown.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[:idx] + lines[idx + 1 :]).strip()
    return markdown.strip()


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as zf:
            if "manifest.json" not in zf.namelist():
                return None
            return json.loads(zf.read("manifest.json").decode("utf-8"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return None


def discover_example_bundles(data_dir: str | Path = DATA_DIR) -> list[ExampleBundleInfo]:
    bundles: list[ExampleBundleInfo] = []
    for path in sorted(Path(data_dir).glob("*.zip")):
        manifest = _read_manifest(path)
        if not manifest or manifest.get("format") != AMAZON_CLUSTER_FORMAT:
            continue
        dataset_id = str(manifest.get("dataset_id") or path.stem)
        title = str(manifest.get("title") or dataset_id.replace("_", " "))
        bundles.append(
            ExampleBundleInfo(
                path=path,
                dataset_id=dataset_id,
                title=title,
                format=str(manifest["format"]),
                version=int(manifest.get("version", 1)),
            )
        )
    return bundles


def load_example_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        clustering = manifest.get("stages", {}).get("clustering", {})
        metadata_path = str(clustering.get("metadata_path") or "clustering/meta.feather")
        clusters_path = str(clustering.get("clusters_path") or "clustering/clusters.feather")

        metadata = pd.read_feather(BytesIO(zf.read(metadata_path)))
        clusters = pd.read_feather(BytesIO(zf.read(clusters_path)))

    return {
        "manifest": manifest,
        "metadata": metadata,
        "clusters": clusters,
        "title": str(manifest.get("title") or path.stem.replace("_", " ")),
    }
