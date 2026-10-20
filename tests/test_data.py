"""Tests de la lecture des étiquettes YOLO (étape 2.8)."""
from pathlib import Path

import pandas as pd
import pytest

from src.data import build_annotation_tables, label_consistency, parse_label_file


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_label_file(tmp_path):
    boxes, issues = parse_label_file(write(tmp_path, "a.txt", "1 0.5 0.4 0.2 0.1\n\n3 0.3 0.3 0.05 0.05\n"))
    assert issues == [] and boxes == [(1.0, 0.5, 0.4, 0.2, 0.1), (3.0, 0.3, 0.3, 0.05, 0.05)]


def test_polygon_malformed_and_non_numeric_lines(tmp_path):
    text = "1 0.1 0.1 0.2 0.2 0.3 0.3\n2 0.5 0.5\n0 a b c d\n1 0.5 0.5 0.2 0.2\n"
    boxes, issues = parse_label_file(write(tmp_path, "b.txt", text))
    assert boxes == [(1.0, 0.5, 0.5, 0.2, 0.2)]                   # seules les lignes valides sont gardées
    assert sorted(issues) == ["malformed", "non_numeric", "polygon"]


def test_empty_file_has_no_box(tmp_path):
    assert parse_label_file(write(tmp_path, "c.txt", "")) == ([], [])


def make_inventory(tmp_path):
    files = {
        "ok.txt": "0 0.5 0.5 0.2 0.2\n",
        "out.txt": "0 0.95 0.5 0.2 0.2\n0 0.5 0.5 0.0 0.2\n",       # sort du cadre + taille nulle
        "mixed.txt": "1 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n",
        "wrong.txt": "3 0.5 0.5 0.2 0.2\n",
    }
    rows = []
    folder = {"ok.txt": "Glioma", "out.txt": "Glioma", "mixed.txt": "Meningioma", "wrong.txt": "Glioma"}
    for name, text in files.items():
        p = write(tmp_path, name, text)
        rows.append(dict(label_path=str(p), **{"class": folder[name]}, split_origin="Train", has_label=True))
    return pd.DataFrame(rows)


def test_annotation_flags_and_consistency(tmp_path):
    inv = make_inventory(tmp_path)
    boxes, files = build_annotation_tables(inv)
    name = lambda path: Path(path).name
    flags = {n: ",".join(g["flags"]) for n, g in boxes.groupby(boxes["label_path"].map(name))}
    assert flags["ok.txt"] == ""
    assert "outside_image" in flags["out.txt"] and "zero_size" in flags["out.txt"]
    cons = label_consistency(boxes, files)
    cons = dict(zip(cons["label_path"].map(name), cons["consistency"]))
    assert cons["ok.txt"] == "all_match" and cons["mixed.txt"] == "mixed" and cons["wrong.txt"] == "none_match"