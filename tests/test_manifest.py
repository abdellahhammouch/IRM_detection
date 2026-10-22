"""Tests du manifeste réel (ignorés si le fichier n'existe pas) (étape 2.8)."""
import json

import pytest

from src.manifest import load_manifest, validate_manifest
from src.preprocessing import select_images
from src.utils import ROOT

MANIFEST = ROOT / "manifests" / "manifest.csv"
REPORT = ROOT / "manifests" / "data_report.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifests/manifest.csv absent")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_manifest_is_valid(manifest):
    problems = validate_manifest(manifest, check_files=(ROOT / "data" / "raw").exists())
    assert problems == []


@pytest.mark.skipif(not REPORT.exists(), reason="data_report.json absent")
def test_manifest_matches_report(manifest):
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    assert len(manifest) == rep["n_images"]
    assert int(manifest["use_for_classification"].sum()) == rep["use_for_classification"]
    assert int(manifest["use_for_detection"].sum()) == rep["use_for_detection"]
    assert manifest["group_id"].nunique() == rep["n_groups"]


def test_split_sizes_used_by_the_pipeline(manifest):
    cls = {s: len(select_images(manifest, s)) for s in ("train", "val", "test")}
    det = {s: len(select_images(manifest, s, "detection")) for s in ("train", "val", "test")}
    assert cls == {"train": 2542, "val": 545, "test": 545}
    assert det == {"train": 2291, "val": 491, "test": 491}


def test_no_group_is_split_across_splits(manifest):
    usable = manifest[manifest["use_for_classification"]]
    assert (usable.groupby("group_id")["split"].nunique() == 1).all()