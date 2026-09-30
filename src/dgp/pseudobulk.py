"""Semi-synthetic pseudo-bulk generation from scRNA-seq reference."""
import numpy as np
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class PseudoBulkDataset:
    """Output of pseudo-bulk generation.

    bulk: pseudo-bulk expression matrix (n_samples, n_genes)
    proportions: true cell type proportions (n_samples, K)
    signature: cell-type signature matrix (n_genes, K)
    cell_type_names: list of cell type names
    gene_names: list of gene names
    """
    bulk: np.ndarray
    proportions: np.ndarray
    signature: np.ndarray
    cell_type_names: list[str]
    gene_names: list[str]


def load_scrna_reference(h5ad_path: str, celltype_key: str = "cell_type",
                         min_cells_per_type: int = 50,
                         n_top_genes: int = 2000):
    """Load scRNA-seq reference from h5ad, return expression matrix and labels.

    Args:
        h5ad_path: path to .h5ad file
        celltype_key: obs column with cell type labels
        min_cells_per_type: drop types with fewer cells
        n_top_genes: number of highly variable genes to keep

    Returns:
        expr: expression matrix (n_cells, n_genes), dense, counts or normalized
        labels: cell type labels (n_cells,)
        gene_names: list of gene names
        cell_type_names: list of retained cell types
    """
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    log.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    # Filter cell types with too few cells
    type_counts = adata.obs[celltype_key].value_counts()
    keep_types = type_counts[type_counts >= min_cells_per_type].index.tolist()
    adata = adata[adata.obs[celltype_key].isin(keep_types)].copy()
    log.info(f"Kept {len(keep_types)} cell types, {adata.n_obs} cells")

    # Normalize + select HVGs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Dense matrix
    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float64)

    labels = adata.obs[celltype_key].values
    gene_names = adata.var_names.tolist()
    cell_type_names = sorted(keep_types)

    return expr, labels, gene_names, cell_type_names


def build_signature(expr: np.ndarray, labels: np.ndarray,
                    cell_type_names: list[str]) -> np.ndarray:
    """Build signature matrix: mean expression per cell type.

    Args:
        expr: (n_cells, n_genes)
        labels: (n_cells,)
        cell_type_names: ordered cell type names

    Returns:
        signature: (n_genes, K)
    """
    n_genes = expr.shape[1]
    K = len(cell_type_names)
    sig = np.zeros((n_genes, K))
    for k, ct in enumerate(cell_type_names):
        mask = labels == ct
        sig[:, k] = expr[mask].mean(axis=0)
    return sig


def sample_proportions(n: int, K: int, rng: np.random.Generator,
                       concentration: float = 1.0) -> np.ndarray:
    """Sample random proportions from Dirichlet.

    Args:
        n: number of samples
        K: number of cell types
        rng: numpy random generator
        concentration: Dirichlet concentration.
            <1: sparse/peaky, =1: uniform, >1: balanced

    Returns:
        proportions: (n, K), rows sum to 1
    """
    alpha = np.full(K, concentration)
    return rng.dirichlet(alpha, size=n)


def generate_pseudobulk(
    expr: np.ndarray,
    labels: np.ndarray,
    cell_type_names: list[str],
    gene_names: list[str],
    n_samples: int = 5000,
    cells_per_sample: int = 200,
    concentration: float = 1.0,
    noise_sd: float = 0.1,
    seed: int = 2026,
) -> PseudoBulkDataset:
    """Generate pseudo-bulk dataset with known ground-truth proportions.

    For each sample:
    1. Draw proportions from Dirichlet(concentration)
    2. Sample cells_per_sample cells according to proportions
    3. Sum their expression to get pseudo-bulk
    4. Add optional Gaussian noise

    Args:
        expr: scRNA-seq expression (n_cells, n_genes)
        labels: cell type labels (n_cells,)
        cell_type_names: ordered cell type names
        gene_names: gene names
        n_samples: number of pseudo-bulk samples
        cells_per_sample: cells mixed per sample
        concentration: Dirichlet parameter
        noise_sd: Gaussian noise on log-expression (0 = no noise)
        seed: random seed

    Returns:
        PseudoBulkDataset
    """
    rng = np.random.default_rng(seed)
    K = len(cell_type_names)
    n_genes = expr.shape[1]

    # Index cells by type
    type_indices = {}
    for k, ct in enumerate(cell_type_names):
        type_indices[ct] = np.where(labels == ct)[0]

    # Sample proportions
    props = sample_proportions(n_samples, K, rng, concentration)

    # Generate pseudo-bulk
    bulk = np.zeros((n_samples, n_genes))
    actual_props = np.zeros((n_samples, K))

    for i in range(n_samples):
        # Number of cells per type (multinomial)
        counts = rng.multinomial(cells_per_sample, props[i])
        actual_props[i] = counts / counts.sum()

        # Sample and sum cells
        sample_expr = np.zeros(n_genes)
        for k, ct in enumerate(cell_type_names):
            if counts[k] > 0:
                idx = rng.choice(type_indices[ct], size=counts[k], replace=True)
                sample_expr += expr[idx].sum(axis=0)

        bulk[i] = sample_expr / cells_per_sample

    # Optional noise
    if noise_sd > 0:
        bulk = bulk + rng.normal(0, noise_sd, bulk.shape)
        bulk = np.maximum(bulk, 0)

    # Build signature
    sig = build_signature(expr, labels, cell_type_names)

    log.info(f"Generated {n_samples} pseudo-bulk samples, {K} types, {n_genes} genes")

    return PseudoBulkDataset(
        bulk=bulk,
        proportions=actual_props,
        signature=sig,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
    )
