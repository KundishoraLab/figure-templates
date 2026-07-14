"""Synthetic data generators for the gallery.

Every generator is seeded, so the gallery is reproducible. None of this is
real data — it exists so the plots can be exercised end-to-end without any
private input. The shapes and column names mirror what the real tools emit
(DESeq2, fgsea, decoupleR, scanpy), so swapping in real data is a path
change, not a rewrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 0

GENE_POOL = [
    "SOX9", "POSTN", "COL1A1", "COL3A1", "ACTA2", "MYH11", "PECAM1", "CDH5",
    "VWF", "KDR", "FLT1", "NOTCH4", "HEY2", "EFNB2", "NR2F2", "NRP2", "EMCN",
    "PLVAP", "ESM1", "APLN", "ANGPT2", "DLL4", "JAG1", "HES1", "ID1", "ID3",
    "SERPINE1", "THBS1", "FN1", "SPARC", "TAGLN", "CNN1", "PDGFRB", "RGS5",
    "KCNJ8", "ABCC9", "GJA4", "GJA5", "SEMA3G", "BMX", "VEGFA", "VEGFC",
    "HIF1A", "EPAS1", "MKI67", "TOP2A", "CCND1", "MYC", "JUN", "FOS", "EGR1",
    "KLF2", "KLF4", "NFKB1", "RELA", "STAT1", "STAT3", "IRF1", "TNF", "IL6",
    "ICAM1", "VCAM1", "SELE", "CXCL8", "CCL2", "MMP2", "MMP9", "TIMP1",
]

HALLMARKS = [
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_HYPOXIA",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_ANGIOGENESIS",
    "HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_KRAS_SIGNALING_UP",
    "HALLMARK_KRAS_SIGNALING_DN", "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_APOPTOSIS", "HALLMARK_P53_PATHWAY", "HALLMARK_GLYCOLYSIS",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_NOTCH_SIGNALING", "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_TGF_BETA_SIGNALING", "HALLMARK_COMPLEMENT",
    "HALLMARK_COAGULATION",
]

TFS = ["RELA", "NFKB1", "JUN", "FOS", "STAT1", "STAT3", "MYC", "E2F1", "SP1",
       "KLF2", "KLF4", "HIF1A", "EPAS1", "SOX17", "SOX18", "ERG", "FLI1",
       "GATA2", "TCF4", "LEF1", "SMAD3", "TEAD1", "FOXO1", "ETS1", "SRF",
       "NR2F2", "HEY2", "HES1", "MEF2C", "PRDM1"]

CELL_TYPES = ["Astrocytes", "Neurons", "Oligodendrocytes",
              "Microglia and Macrophages", "T cells", "Fibroblasts",
              "Pericytes", "Smooth muscle cells"]

EC_SUBTYPES = ["Large artery", "Artery", "Arteriole", "Capillary",
               "Angiogenic capillary", "Venule", "Vein", "Large vein"]

SAMPLES = ["AVM_01", "AVM_04", "AVM_05", "AVM_06"]


def de_table(n_genes: int = 2000, n_real: int = 60, seed: int = SEED,
             include_noise: bool = True) -> pd.DataFrame:
    """DESeq2-shaped DE table: gene, log2FoldChange, pvalue, padj, baseMean.

    Most genes are null; `n_real` carry a true effect. Includes noise-class
    names (MT-, LINC, RPL) and control probes so the exclude-pattern
    machinery has something to actually exclude.
    """
    rng = np.random.RandomState(seed)
    names = list(GENE_POOL)
    i = 0
    while len(names) < n_genes:
        names.append(f"GENE{i:04d}")
        i += 1
    names = names[:n_genes]

    if include_noise:
        noise = (
            [f"MT-ND{k}" for k in range(1, 7)] +
            [f"RPL{k}" for k in range(1, 11)] +
            [f"RPS{k}" for k in range(1, 8)] +
            [f"LINC{1000 + k}" for k in range(8)] +
            [f"MIR{100 + k}" for k in range(4)] +
            [f"AL{100000 + k}.1" for k in range(2)] +
            ["Negative01", "Negative02", "SystemControl1", "FalseCode3"]
        )
        # Slice by len(noise), not a literal — a hardcoded width silently
        # changes len(names) the moment the noise list grows.
        names[-len(noise):] = noise

    lfc = rng.normal(0, 0.35, n_genes)
    base = 10 ** rng.uniform(0.5, 3.5, n_genes)
    real = rng.choice(n_genes, size=min(n_real, n_genes), replace=False)
    lfc[real] += rng.choice([-1, 1], n_real) * rng.uniform(0.8, 3.2, n_real)

    # p from a Wald test with a depth-dependent standard error: low-count
    # genes get a big SE and stay non-significant however large their fold
    # change, which is the real shape of a DE volcano (the classic
    # low-expression high-|lfc| skirt along the bottom). Deriving p from an
    # effect/SE ratio rather than sampling it directly also means a few null
    # genes cross threshold by chance, as they must.
    se = 0.25 + 1.5 / np.sqrt(base)
    p = _two_sided_p(lfc / se)

    order = np.argsort(p)
    padj = np.empty(n_genes)
    ranks = np.arange(1, n_genes + 1)
    padj[order] = np.minimum.accumulate((p[order] * n_genes / ranks)[::-1])[::-1]
    padj = np.clip(padj, 0, 1)

    # DESeq2 sets padj = NaN for independent-filtered low-count genes.
    low = base < 15
    padj[low] = np.nan

    return pd.DataFrame({
        "gene": names, "baseMean": base, "log2FoldChange": lfc,
        "pvalue": p, "padj": padj,
    })


def pathway_table(seed: int = SEED) -> pd.DataFrame:
    """fgsea-shaped: pathway, NES, pval, padj, size."""
    rng = np.random.RandomState(seed + 1)
    n = len(HALLMARKS)
    nes = rng.normal(0, 1.4, n)
    nes[:5] += rng.uniform(0.8, 1.8, 5)      # a few strong up
    nes[-4:] -= rng.uniform(0.8, 1.6, 4)     # a few strong down
    p = np.clip(10 ** (-np.abs(nes) * rng.uniform(1.0, 2.6, n)), 1e-30, 1)
    padj = np.clip(p * n / np.arange(1, n + 1)[np.argsort(np.argsort(p))], 0, 1)
    return pd.DataFrame({
        "pathway": HALLMARKS, "NES": nes, "pval": p, "padj": padj,
        "size": rng.randint(15, 400, n),
    })


def tf_table(seed: int = SEED) -> pd.DataFrame:
    """decoupleR-shaped: source, delta (activity change), q."""
    rng = np.random.RandomState(seed + 2)
    n = len(TFS)
    delta = rng.normal(0, 1.0, n)
    delta[:6] += rng.uniform(1.0, 2.5, 6)
    delta[-5:] -= rng.uniform(1.0, 2.2, 5)
    q = np.clip(10 ** (-np.abs(delta) * rng.uniform(0.8, 3.0, n)), 1e-25, 1)
    return pd.DataFrame({"source": TFS, "delta": delta, "q": q,
                         "n_targets": rng.randint(10, 500, n)})


def embedding(n_cells: int = 6000, seed: int = SEED):
    """UMAP-shaped coords + obs frame (cell_type, ec_subtype, sample, score).

    Clusters are drawn per cell type so the embedding has real structure
    rather than one blob — otherwise the categorical UMAP demo shows nothing.
    """
    rng = np.random.RandomState(seed + 3)
    labels = EC_SUBTYPES + CELL_TYPES
    k = len(labels)

    # Cluster centers on a ring, so neighbors are visually adjacent.
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = np.c_[np.cos(ang), np.sin(ang)] * 7.0
    centers += rng.normal(0, 0.5, centers.shape)

    weights = rng.dirichlet(np.ones(k) * 2.5)
    counts = rng.multinomial(n_cells, weights)

    xy, lab = [], []
    for i, (c, n) in enumerate(zip(centers, counts)):
        if n == 0:
            continue
        xy.append(rng.normal(c, rng.uniform(0.7, 1.5), size=(n, 2)))
        lab += [labels[i]] * n
    xy = np.vstack(xy)
    lab = np.array(lab)

    perm = rng.permutation(len(lab))
    xy, lab = xy[perm], lab[perm]

    is_ec = np.isin(lab, EC_SUBTYPES)
    obs = pd.DataFrame({
        "label": lab,
        "cell_type": np.where(is_ec, "Endothelial", lab),
        "ec_subtype": np.where(is_ec, lab, "NA"),
        "sample": rng.choice(SAMPLES, len(lab), p=[0.3, 0.3, 0.25, 0.15]),
        # Score elevated in venous EC, so the continuous demo shows a gradient.
        "score": rng.normal(0, 1, len(lab)) +
                 np.isin(lab, ["Venule", "Vein", "Large vein"]) * 2.2,
    })
    obs["is_mutant"] = (rng.uniform(size=len(lab)) <
                        np.where(obs["score"] > 1.5, 0.25, 0.02))
    return xy, obs


def spatial(n_cells: int = 4000, seed: int = SEED):
    """Tissue-shaped coords in mm + obs, for the spatial-map demo."""
    rng = np.random.RandomState(seed + 4)
    # A vessel-like band plus diffuse parenchyma.
    n_vessel = n_cells // 3
    t = rng.uniform(0, 1, n_vessel)
    vx = 0.5 + 3.0 * t
    vy = 1.6 + 0.6 * np.sin(t * 5) + rng.normal(0, 0.08, n_vessel)
    vessel = np.c_[vx, vy]
    paren = rng.uniform([0, 0], [4, 3], size=(n_cells - n_vessel, 2))
    xy = np.vstack([vessel, paren])
    lab = np.array(["Vein"] * n_vessel +
                   list(rng.choice(CELL_TYPES, n_cells - n_vessel)))
    d = np.abs(xy[:, 1] - (1.6 + 0.6 * np.sin(
        np.clip((xy[:, 0] - 0.5) / 3.0, 0, 1) * 5)))
    obs = pd.DataFrame({
        "label": lab, "dist_to_vessel_mm": d,
        "score": np.exp(-d * 1.8) * 3 + rng.normal(0, 0.35, n_cells),
    })
    return xy, obs


def composition(seed: int = SEED) -> pd.DataFrame:
    """Per-sample x cell-type count matrix."""
    rng = np.random.RandomState(seed + 5)
    cats = EC_SUBTYPES + CELL_TYPES[:4]
    rows = {s: rng.multinomial(rng.randint(600, 2500), rng.dirichlet(np.ones(len(cats)) * 1.6))
            for s in SAMPLES}
    return pd.DataFrame(rows, index=cats).T


def expression_matrices(genes, groups=("proximal", "distal"), seed: int = SEED):
    """(mean_df, pct_df) for dotplot_expression, without needing an AnnData."""
    rng = np.random.RandomState(seed + 6)
    genes = list(genes)
    mean = rng.gamma(2.0, 0.5, size=(len(groups), len(genes)))
    pct = np.clip(rng.beta(2, 2, size=(len(groups), len(genes))) * 100, 2, 99)
    # Give the first group a real lift on the first half of the genes, so the
    # panel shows contrast rather than noise.
    half = len(genes) // 2
    mean[0, :half] *= rng.uniform(1.6, 3.0, half)
    pct[0, :half] = np.clip(pct[0, :half] * rng.uniform(1.2, 1.8, half), 2, 99)
    if len(groups) > 1:
        mean[1, half:] *= rng.uniform(1.6, 3.0, len(genes) - half)
        pct[1, half:] = np.clip(pct[1, half:] * rng.uniform(1.2, 1.8,
                                                            len(genes) - half), 2, 99)
    return (pd.DataFrame(mean, index=list(groups), columns=genes),
            pd.DataFrame(pct, index=list(groups), columns=genes))


def score_matrix(seed: int = SEED) -> pd.DataFrame:
    """Pathway x cell-type signed score matrix, for the heatmap demo."""
    rng = np.random.RandomState(seed + 7)
    rows = [p.replace("HALLMARK_", "").replace("_", " ").title()
            for p in HALLMARKS[:12]]
    m = rng.normal(0, 1.0, size=(len(rows), len(EC_SUBTYPES)))
    m[:4, 5:] += 1.8   # a venous-biased block
    m[8:, :3] -= 1.5   # an arterial-depleted block
    df = pd.DataFrame(m, index=rows, columns=EC_SUBTYPES)
    df.iloc[2, 6] = np.nan   # exercise the NA-grey path
    return df


def interaction_matrix(seed: int = SEED) -> pd.DataFrame:
    """Square sender x receiver signed delta matrix, for the chord demo."""
    rng = np.random.RandomState(seed + 8)
    cats = EC_SUBTYPES[:4] + CELL_TYPES[:4]
    m = rng.normal(0, 1, size=(len(cats), len(cats)))
    m[np.abs(m) < 0.8] = 0
    np.fill_diagonal(m, 0)
    return pd.DataFrame(m, index=cats, columns=cats)


def gene_sets(seed: int = SEED) -> dict:
    """Named overlapping gene sets, for the upset demo."""
    rng = np.random.RandomState(seed + 9)
    pool = GENE_POOL
    return {
        "scRNA": set(rng.choice(pool, 34, replace=False)),
        "CosMx": set(rng.choice(pool, 30, replace=False)),
        "Bulk": set(rng.choice(pool, 26, replace=False)),
        "Public": set(rng.choice(pool, 22, replace=False)),
    }


def _two_sided_p(z):
    """Two-sided normal p-value from a z-score.

    Uses erfc directly rather than `1 - erf(...)`: the latter underflows to
    exactly 0 for |z| beyond ~8.2 in float64, which would pin every strong
    gene to the clip floor and draw a fake horizontal stripe across the top
    of the volcano. erfc stays accurate into the 1e-300s.
    """
    from math import erfc, sqrt
    z = np.asarray(z, dtype=float)
    flat = [erfc(abs(v) / sqrt(2)) for v in z.ravel()]
    return np.clip(np.array(flat).reshape(z.shape), 1e-300, 1.0)
