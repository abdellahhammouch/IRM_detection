import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import ROOT

HASH_W, HASH_H = 16, 16   # empreinte visuelle de 16 x 16 = 256 oui/non
THUMB = 32                # miniature 32 x 32 pour comparer les pixels


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _visual_hash(gray: Image.Image) -> np.ndarray:
    """Réduit l'image à 17x16 pixels gris, puis note si chaque pixel est plus clair
    que son voisin de droite : 256 valeurs oui/non."""
    small = np.asarray(gray.resize((HASH_W + 1, HASH_H), Image.BILINEAR), dtype=np.float32)
    return (small[:, 1:] > small[:, :-1]).flatten()


def compute_fingerprints(paths, cache_path=None):
    """Retourne (tableau sha256, empreintes visuelles, miniatures), alignés sur 'paths'."""
    paths = list(paths)
    if cache_path and Path(cache_path).exists():
        z = np.load(cache_path)
        if list(z["paths"]) == paths:
            return pd.DataFrame({"image_path": paths, "sha256": z["sha256"]}), z["bits"], z["thumbs"]
    sha, bits, thumbs = [], [], []
    for rel in paths:
        p = ROOT / rel
        sha.append(_sha256(p))
        with Image.open(p) as im:
            gray = im.convert("L")
        bits.append(_visual_hash(gray))
        thumbs.append(np.asarray(gray.resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8))
    bits, thumbs = np.array(bits, dtype=bool), np.array(thumbs, dtype=np.uint8)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, paths=np.array(paths), sha256=np.array(sha),
                            bits=bits, thumbs=thumbs)
    return pd.DataFrame({"image_path": paths, "sha256": sha}), bits, thumbs


def summarize_groups(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Pour chaque groupe : nombre de fichiers, de classes, de splits."""
    return df.groupby(key).agg(
        n=("image_path", "size"),
        n_classes=("class", "nunique"),
        n_splits=("split_origin", "nunique"),
    )


def nearest_distance(bits, chunk=1000):
    """Pour chaque image : distance (en bits différents) à l'image la plus proche."""
    X = bits.astype(np.float32)
    Xn, (n, d) = 1.0 - X, X.shape
    nearest = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dist = d - (X[s:e] @ X.T + Xn[s:e] @ Xn.T)
        dist[np.arange(e - s), np.arange(s, e)] = d + 1   # ignorer l'image elle-même
        nearest[s:e] = dist.min(axis=1)
    return nearest


def hamming_pairs(bits, max_dist, chunk=1000) -> pd.DataFrame:
    """Toutes les paires (i, j) dont la distance est <= max_dist."""
    X = bits.astype(np.float32)
    Xn, (n, d) = 1.0 - X, X.shape
    out = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dist = d - (X[s:e] @ X.T + Xn[s:e] @ Xn.T)
        rows, cols = np.nonzero(dist <= max_dist + 0.5)
        keep = (rows + s) < cols                       # chaque paire une seule fois
        out.append(np.column_stack([rows[keep] + s, cols[keep], dist[rows[keep], cols[keep]]]))
    arr = np.vstack(out) if out else np.empty((0, 3))
    return pd.DataFrame(arr.astype(int), columns=["i", "j", "hamming"])


def describe_pairs(pairs, meta, thumbs, chunk=20000) -> pd.DataFrame:
    """Ajoute les infos des deux images et l'écart moyen de pixels (mad, de 0 à 255)."""
    p = pairs.copy()
    for side, col in (("a", "i"), ("b", "j")):
        m = meta.iloc[p[col].values].reset_index(drop=True)
        p[f"path_{side}"] = m["image_path"]
        p[f"class_{side}"] = m["class"]
        p[f"split_{side}"] = m["split_origin"]
    p["same_class"] = p.class_a == p.class_b
    p["same_split"] = p.split_a == p.split_b
    mad = []
    for s in range(0, len(p), chunk):
        i, j = p.i.values[s:s + chunk], p.j.values[s:s + chunk]
        mad.append(np.abs(thumbs[i].astype(np.int16) - thumbs[j].astype(np.int16)).mean(axis=(1, 2)))
    p["mad"] = np.round(np.concatenate(mad), 2) if mad else []
    return p


def plot_pairs(pp, per_row=3, size=2.4, out_path=None):
    """Affiche chaque paire côte à côte, avec distance (ham) et écart de pixels (mad)."""
    n = len(pp)
    nrows = -(-n // per_row)
    fig, axes = plt.subplots(nrows, per_row * 2, figsize=(per_row * 2 * size, nrows * size * 1.15))
    axes = np.array(axes).reshape(nrows, per_row * 2)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(pp.iterrows()):
        row, col = divmod(k, per_row)
        for off, side in enumerate("ab"):
            ax = axes[row, col * 2 + off]
            ax.imshow(Image.open(ROOT / r[f"path_{side}"]).convert("L"), cmap="gray")
            title = f"{r[f'class_{side}']} | {Path(r[f'path_{side}']).stem}"
            if off == 0:
                title += f"\nham={r.hamming}  mad={r.mad}"
                if "corr" in r.index:
                    title += f"  corr={r['corr']}"
            ax.set_title(title, fontsize=6)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()

def build_groups(n, pairs):
    """Un numéro de groupe par image : deux images liées par une paire sont dans le même groupe."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(pairs)), (pairs["i"], pairs["j"])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels
    


def add_correlation(pairs, thumbs, chunk=20000):
    """Ajoute 'corr' : ressemblance entre deux miniatures qui ignore la luminosité.
    1.0 = même image à la luminosité et au contraste près."""
    t = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    t = t - t.mean(axis=1, keepdims=True)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    out = pairs.copy()
    c = []
    for s in range(0, len(out), chunk):
        i, j = out.i.values[s:s + chunk], out.j.values[s:s + chunk]
        c.append((t[i] * t[j]).sum(axis=1))
    out["corr"] = np.round(np.concatenate(c), 3) if c else []
    return out

def select_twins(pp, ham_max=20, corr_min=0.99, mad_max=2.0):
    """Paires jumelles : proches par la distance, ET (très ressemblantes à la
    luminosité près, OU pixels quasi égaux)."""
    close = pp["hamming"] <= ham_max
    similar = (pp["corr"] >= corr_min) | (pp["mad"] <= mad_max)
    return pp[close & similar]


def group_diagnostics(meta, thumbs, min_size=8):
    """Pour les gros groupes : ressemblance minimale (corr) entre deux membres.
    Un corr minimum bas signale un effet de chaîne."""
    t = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    t = t - t.mean(axis=1, keepdims=True)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    sizes = meta.groupby("group_id").size()
    rows = []
    for gid in sizes[sizes >= min_size].sort_values(ascending=False).index:
        idx = np.flatnonzero(meta.group_id.values == gid)
        corr = t[idx] @ t[idx].T
        rows.append({"group_id": gid, "n": len(idx), "classe": meta["class"].iloc[idx[0]],
                     "corr minimum": round(float(corr.min()), 3)})
    return pd.DataFrame(rows)