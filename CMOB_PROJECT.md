# CMOB Project: Multi-Omics Cancer Classification with Soft Permutation Mixing
> Author: Ward Abdelhafez  
> Started: February 23, 2026  
> Last Updated: February 24, 2026  
> Environment: MacBook Pro M1 · conda env `jlab` · PyTorch 2.10.0 · MPS enabled  
> Dataset: CMOB — Cancer Multi-Omics Benchmark (TCGA Pan-Cancer + GS-BRCA)  
> GitHub: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark

---

## Project Summary

This project applies a **SoftPermutationMix** layer — inspired by DeepSeek's mHC-lite
architecture (arXiv:2601.05732, Jan 2026) — to multi-omics cancer classification using
the CMOB benchmark dataset. The goal is to:

1. Implement and validate the doubly stochastic mixing layer from mHC-lite.
2. Apply it to TCGA multi-omics data (mRNA, miRNA, methylation, CNV).
3. Benchmark against a vanilla baseline (concatenation + MLP).
4. Interpret the **cross-omics information flow** learned by the mixing matrix.
5. Contribute novel biological insight to a dataset currently under community review.

---

## Theoretical Background

### 1. Residual Connections to Hyper-Connections to mHC

| Year | Paper | Key idea |
|------|-------|----------|
| 2017 | Attention Is All You Need (Transformer) | Residual skip: x → x + f(x) stabilizes deep training |
| 2024 | Hyper-Connections (HC) | Multiple residual streams + learned mixing matrix |
| Dec 2025 | DeepSeek mHC | Manifold-constrained HC: mixing matrices projected onto Birkhoff polytope |
| Jan 2026 | mHC-lite (arXiv:2601.05732) | Same constraint, implemented via convex combination of permutations (no Sinkhorn) |

### 2. Weighted Sum of Permutations (the core math)

A permutation matrix reorders a vector with no scaling. A convex combination of K
permutation matrices:

    D = a1*P1 + a2*P2 + ... + aK*PK,   ai >= 0,   sum(ai) = 1

is guaranteed to be doubly stochastic: every row and column sums to 1.

Birkhoff-von Neumann theorem: Every doubly stochastic matrix is a convex combination
of permutation matrices. The set of all doubly stochastic matrices = the Birkhoff polytope.

Why this matters for deep learning:
- Doubly stochastic matrices have operator norm <= 1, so no signal explosion.
- Parameterizing D this way makes the constraint exact by construction (no iterative
  normalization needed, unlike original mHC which requires 20 Sinkhorn-Knopp iterations).
- Networks can learn which permutations to mix (via softmax weights alpha), giving
  structured, interpretable routing of information.

Analogy: A single permutation = one perfect shuffle of a deck. A weighted sum =
"60% shuffle A, 40% shuffle B, averaged" — a soft, probabilistic re-routing.

### 3. From mHC to Multi-Omics

In the original mHC paper, the doubly stochastic constraint is applied to residual
stream mixing in ultra-deep LLMs to prevent training collapse (max gain magnitude
reduced from ~3000x to ~1.6x). In this project, the same constraint is applied to
cross-omics feature routing:

- Each omics block (mRNA, miRNA, methylation, CNV) is independently encoded.
- A SoftPermutationMix layer mixes the concatenated latent representations.
- The learned mixing matrix reveals which omics modalities inform which others.

---

## Dataset: CMOB

| Property | Value |
|----------|-------|
| Source | TCGA Pan-Cancer |
| Patients | 8,314 |
| Cancer types | 32 |
| Omics blocks | mRNA, miRNA, Methylation, CNV |
| Tasks | Classification, subtyping, imputation (20 total) |
| Status | Published as MLOmics in Nature Scientific Data (May 2025) |
| GitHub | https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark |
| Figshare | https://figshare.com/articles/dataset/MLOmics/28729127 |

### Real Data Dimensions (Original Pan-cancer, post-transpose)

| Omics block | Features | Patients |
|-------------|----------|---------|
| mRNA        | 3,217    | 8,314   |
| miRNA       | 383      | 8,314   |
| Methylation | 3,139    | 8,314   |
| CNV         | 3,105    | 8,314   |
| **Total**   | **9,844**| **8,314**|

### Data Loading Notes (IMPORTANT)
- CSVs are stored as (features × patients) — must transpose with `.T` on load
- Label file (`Pan-cancer_label_num.csv`) has no patient barcodes — positional alignment
- Load labels WITHOUT index_col: `pd.read_csv(...)`  (not index_col=0)
- All missing values pre-imputed by dataset authors (0 NaNs in Original version)
- Preprocessing: median imputation + StandardScaler z-score normalization per feature

### Why CMOB benefits from SoftPermMix

1. Heterogeneous scales: Methylation has ~450K features vs ~1.8K for miRNA.
   Naive concatenation lets methylation numerically dominate. Mass-preserving mixing corrects this.
2. Known biological cross-talk: CNV drives mRNA expression; miRNA represses mRNA;
   methylation silences genes. The mixing matrix learns these dependencies from data.
3. High dimensionality / small n: 650K+ features, 8K patients — deep models collapse
   without norm control. Doubly stochastic constraint prevents this.
4. CMOB benchmark gap: Authors explicitly state "deep methods have significant room
   for improvement" and existing baselines are simple concatenation.

### Expected cross-omics flow patterns (biological ground truth)

    Target     Source
               mRNA   miRNA  Methy  CNV
    mRNA       high   med    low    med    <- CNV amplifies, miRNA represses
    miRNA      low    high   low    low    <- Mostly self-contained
    Methy      low    low    high   med    <- CNV influences epigenome
    CNV        low    low    low    high   <- DNA-level, largely independent

---

## Environment Setup

### Conda env: jlab

    conda activate jlab
    conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
    conda deactivate && conda activate jlab

### Known issue: libomp conflict on Apple M1
- Symptom: Jupyter kernel dies immediately on "import torch". Terminal gives
  OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
- Cause: pip-installed PyTorch + conda-installed numpy ship conflicting libomp copies.
- Fix: conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE (permanent per env).
- Verify: python -c "import torch; print(torch.backends.mps.is_available())" -> True

### Known issue: MPS register_buffer device mismatch
- Symptom: RuntimeError: Placeholder storage has not been allocated on MPS device!
- Cause: mixer.alpha_logits.data = tensor(...) replaces MPS storage with CPU tensor.
- Fix A: Always call .to(device) after instantiating SoftPermutationMix or MultiOmicsNet.
- Fix B: Use mixer.alpha_logits.copy_(tensor(...).to(device)) for in-place updates.
- Rule: Every standalone nn.Module instantiation must be followed by .to(device).

### PyTorch version
    PyTorch : 2.10.0
    MPS     : True (M1 GPU acceleration active)

### Verified library versions (Feb 23, 2026)
    NumPy        : 2.4.2
    Pandas       : 3.0.0
    Scikit-learn : 1.8.0
    Matplotlib   : 3.10.8
    Seaborn      : 0.13.2
    SciPy        : 1.17.0

### First cell of every notebook

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | Device: {device}")

---

## Notebook Structure

    CMOB/
    ├── 00_environment_check.ipynb      <- verify torch, MPS, all imports
    ├── 01_simulate_and_explore.ipynb   <- simulate CMOB-style data, EDA, PCA
    ├── 02_softperm_module.ipynb        <- SoftPermutationMix unit tests
    ├── 03_model_and_training.ipynb     <- full model, training, baseline comparison
    ├── 04_interpret_results.ipynb      <- mixing matrix, cross-omics flow heatmap
    ├── 05_real_cmob_swap.ipynb         <- load real TCGA CMOB CSVs, re-run
    ├── 06_brca_subtype.ipynb           <- BRCA PAM50 subtype experiment (Phase 2)
    │
    ├── cmob_simulated.csv              <- generated by 01
    ├── cmob_real.csv                   <- generated by 05 (8314 x 9845)
    ├── model_base.pt                   <- saved by 03 (no mixing, Pan-cancer)
    ├── model_mix.pt                    <- saved by 03 (with SoftPermMix, Pan-cancer)
    ├── model_base_brca.pt              <- saved by 06 (no mixing, BRCA)
    ├── model_mix_brca.pt               <- saved by 06 (with SoftPermMix, BRCA)
    │
    └── figures/
        ├── 01_missing_values.png
        ├── 01_pca_overview.png
        ├── 02_mixing_matrix_evolution.png
        ├── 03_training_curves.png           <- Pan-cancer (saturated ~97.7%)
        ├── 04_mixing_matrix_full.png        <- Pan-cancer (uniform alpha)
        ├── 04_crossomics_flow.png           <- Pan-cancer (flat, no signal)
        ├── 06_brca_training_curves.png      <- BRCA subtype (expected ~70-85%)
        └── 06_brca_crossomics_flow.png      <- BRCA subtype (expected structured)

### Notebook descriptions

| Notebook | Purpose | Status | Key outputs |
|----------|---------|--------|-------------|
| 00 | Env check, MPS smoke test | ✅ DONE | MPS=True confirmed |
| 01 | Simulate 300-patient CMOB, EDA, PCA | ✅ DONE | cmob_simulated.csv, 2 figures |
| 02 | Define SoftPermMix, verify doubly stochastic | ✅ DONE (MPS fix applied) | 1 figure |
| 03 | Train baseline vs. SoftPermMix | ✅ DONE (real data) | model_*.pt, training curves |
| 04 | Interpret cross-omics flow | ✅ DONE (real data) | mixing matrix figures |
| 05 | Load real TCGA data, align patients | ✅ DONE | cmob_real.csv (8314×9845) |
| 06 | BRCA PAM50 subtype experiment | 🔄 IN PROGRESS | TBD |

---

## Phase 1 Results: Pan-Cancer Classification (32 classes)

### Actual Results on Real TCGA Data

| Metric | Baseline | + SoftPermMix | Notes |
|--------|----------|---------------|-------|
| Test accuracy (epoch 80) | 97.78% | 97.71% | Both saturated near ceiling |
| Training stability | Dip to 96.6% ~epoch 40 | Smoother convergence | Norm control visible |
| Learned α weights | N/A | [0.2497, 0.2464, 0.2435, 0.2604] | Near-uniform — mixer not engaged |
| Cross-omics flow | N/A | Flat ~0.004 across all pairs | No detectable structure |

### Why Pan-Cancer Saturated

Pan-cancer classification is too easy for the SoftPermMix to show its advantage:
- 32 cancer types have extremely distinct molecular signatures across all 4 omics blocks
- Per-omics encoders independently solve the task before signal reaches the mixer
- Gradient signal to alpha_logits ≈ 0 → weights stay near initialization [0.25, 0.25, 0.25, 0.25]
- This is NOT a bug — it is a property of the task difficulty

The SoftPermMix advantage requires a task where cross-omics interactions are necessary,
not just parallel single-omics signals. Pan-cancer does not require this.

### Notable Observation
SoftPermMix showed the predicted stability advantage: baseline accuracy dipped to ~96.6%
around epoch 40 (the loss spike predicted in design), while SoftPermMix maintained
smoother convergence. This confirms norm control is working, even if accuracy gap is small.

---

## Phase 2: BRCA PAM50 Subtype Classification

### Motivation

BRCA subtyping (PAM50: Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like)
is a fundamentally harder task where cross-omics integration is biologically necessary:

- mRNA alone achieves ~85-90% (PAM50 is gene-expression defined), but subtypes overlap
- Methylation → mRNA: strong (promoter hypermethylation defines Luminal vs Basal identity)
- CNV → mRNA: strong (HER2-enriched has chr17q12 amplification → ERBB2 overexpression)
- miRNA → mRNA: moderate (miR-21 upregulated in Basal; miR-155 in HER2-enriched)
- The mixer MUST learn cross-omics routing to distinguish borderline LumA/LumB cases

### Expected Biological Cross-Omics Flow for BRCA Subtypes

    Target     Source
               mRNA   miRNA  Methy  CNV
    mRNA       high   med    HIGH   HIGH   <- Methy silences; CNV(HER2) drives ERBB2
    miRNA      low    high   low    low    <- Mostly self-contained
    Methy      low    low    high   med    <- CNV-driven chromosomal instability
    CNV        low    low    low    high   <- DNA structural, largely independent

Key difference from Pan-cancer: Methy→mRNA and CNV→mRNA should be MUCH stronger
because HER2 amplification and epigenetic silencing are the defining BRCA subtype features.

### Data Source
- Dataset: CMOB GS-BRCA
- Path: ~/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/GS-BRCA/
- Labels: PAM50 subtypes (5 classes: LumA=0, LumB=1, HER2=2, Basal=3, Normal=4)
- Expected patients: ~1,000 BRCA samples

### Expected Results
| Metric | Baseline | + SoftPermMix | Notes |
|--------|----------|---------------|-------|
| Test accuracy | ~75-85% | ~80-88% | Harder task, more room for improvement |
| Learned α weights | N/A | Non-uniform, e.g. [0.41, 0.28, 0.19, 0.12] | Mixer engages |
| Cross-omics flow | N/A | Structured heatmap | CNV→mRNA and Methy→mRNA elevated |

---

## Core Module Definitions

### SoftPermutationMix

    class SoftPermutationMix(nn.Module):
        """
        Learnable doubly stochastic mixing layer.
        Parameterized as D = sum_k(alpha_k * P_k) where P_k are fixed random
        permutation matrices and alpha_k = softmax(logits) are learned weights.
        Guarantees: row sums = col sums = 1 (doubly stochastic) by construction.
        Inspired by: mHC-lite (arXiv:2601.05732, DeepSeek, Jan 2026)
        """
        def __init__(self, dim, K=4):
            super().__init__()
            self.dim = dim
            self.K = K
            self.alpha_logits = nn.Parameter(torch.zeros(K))
            perms = [torch.eye(dim)[torch.randperm(dim)] for _ in range(K)]
            self.register_buffer("perms", torch.stack(perms))

        def get_mixing_matrix(self):
            alpha = torch.softmax(self.alpha_logits, dim=0)
            return torch.einsum("k,kij->ij", alpha, self.perms)

        def forward(self, x):
            return x @ self.get_mixing_matrix().T

        def get_alpha(self):
            return torch.softmax(self.alpha_logits, dim=0).detach().cpu().numpy()

### MultiOmicsNet

    class MultiOmicsNet(nn.Module):
        """
        Architecture:
        1. Per-omics encoders: Linear -> LayerNorm -> GELU -> Dropout (x4 blocks)
        2. Concatenate latents: [mRNA | miRNA | Methy | CNV] -> (B, 256)
        3. SoftPermutationMix(dim=256, K=4) — cross-omics routing
        4. Classification head: Linear -> LayerNorm -> GELU -> Dropout -> Linear(n_classes)

        use_mix=False gives baseline (same architecture without step 3).
        """
        def __init__(
            self,
            n_mrna=200, n_mirna=50, n_methy=200, n_cnv=200,
            latent_dim=64, n_classes=5, use_mix=True, K=4
        ):
            ...

    # Pan-cancer instantiation:
    MultiOmicsNet(n_mrna=3217, n_mirna=383, n_methy=3139, n_cnv=3105,
                  latent_dim=64, n_classes=32, use_mix=True, K=4)

    # BRCA subtype instantiation (TBD — update after loading GS-BRCA):
    MultiOmicsNet(n_mrna=?, n_mirna=?, n_methy=?, n_cnv=?,
                  latent_dim=64, n_classes=5, use_mix=True, K=4)

---

## Debugging Log

| Date | Notebook | Issue | Fix |
|------|----------|-------|-----|
| Feb 23 | NB02 Cell 5 | MPS Placeholder storage error on alpha_logits.data = tensor | Use .copy_(tensor.to(device)) + always call .to(device) after instantiation |
| Feb 23 | NB03 Cell 6 | Training function defined but never called | Cell 6 = definition, Cell 7 = execution. Always check for calling cell. |
| Feb 23 | NB03 Cell 7 | 100% accuracy on simulated data | Expected — simulated data has clean PCA-separable clusters. Real data is the true test. |
| Feb 24 | NB05 Cell 2 | Shape (3217, 8314) — transposed | CSVs stored as features×patients. Fix: add .T on load. |
| Feb 24 | NB05 Cell 2 | Labels (8314, 0) — empty columns | index_col=0 consumed the label column. Fix: remove index_col from labels load. |
| Feb 24 | NB05 Cell 3 | Common patients: 0 | Labels use integer index, not TCGA barcodes. Fix: reset_index(drop=True) on omics, positional concat. |
| Feb 24 | NB04 Cell 3 | Size mismatch loading model_mix.pt | NB04 used default dimensions (simulated). Fix: pass real dims to MultiOmicsNet instantiation. |

---

## GitHub Repository Plan

    CMOB/
    ├── README.md                        <- project overview + key results
    ├── CMOB_PROJECT.md                  <- this file (detailed lab notebook)
    ├── .gitignore                       <- excludes *.pt, *.csv, __pycache__
    ├── notebooks/
    │   ├── 00_environment_check.ipynb
    │   ├── 01_simulate_and_explore.ipynb
    │   ├── 02_softperm_module.ipynb
    │   ├── 03_model_and_training.ipynb
    │   ├── 04_interpret_results.ipynb
    │   ├── 05_real_cmob_swap.ipynb
    │   └── 06_brca_subtype.ipynb
    ├── src/
    │   ├── model.py                     <- MultiOmicsNet class
    │   └── softperm.py                  <- SoftPermutationMix class
    └── figures/
        ├── pan_cancer_training_curves.png
        ├── pan_cancer_crossomics_flow.png
        ├── brca_training_curves.png
        └── brca_crossomics_flow.png

### .gitignore contents
    *.pt
    *.csv
    __pycache__/
    .ipynb_checkpoints/
    .DS_Store
    *.pyc

---

## Next Steps

- [x] Run notebooks 00 through 04 on simulated data — confirm all outputs
- [x] git clone https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark
- [x] Run notebook 05 to align real TCGA data and generate cmob_real.csv
- [x] Re-run 03 and 04 on real Pan-cancer data
- [ ] Download GS-BRCA data from Figshare
- [ ] Create notebook 06_brca_subtype.ipynb
- [ ] Train on BRCA subtypes — confirm non-uniform alpha weights
- [ ] Interpret BRCA cross-omics flow heatmap biologically
- [ ] Initialize GitHub repo and push notebooks + figures
- [ ] Write README.md with results summary
- [ ] Explore: does flow matrix change per cancer subtype?

---

## References

| Reference | Link |
|-----------|------|
| mHC-lite (arXiv:2601.05732) | https://arxiv.org/abs/2601.05732 |
| CMOB/MLOmics benchmark (arXiv:2409.02143) | https://arxiv.org/abs/2409.02143 |
| MLOmics Nature Sci Data (2025) | https://www.nature.com/articles/s41597-025-05235-x |
| CMOB GitHub | https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark |
| MLOmics Figshare | https://figshare.com/articles/dataset/MLOmics/28729127 |
| Birkhoff-von Neumann theorem | https://en.wikipedia.org/wiki/Doubly_stochastic_matrix |
| PAM50 multi-omics BRCA (2020) | https://www.frontiersin.org/articles/10.3389/fonc.2020.00845 |
| Multi-omics integration survey | https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1425456 |

---

## Conversation Context (for resuming in a new chat)

This project originated in a discussion about DeepSeek mHC (Manifold-Constrained
Hyper-Connections, Dec 2025 arXiv) and its follow-up mHC-lite (Jan 2026, arXiv:2601.05732).

Core concepts established:
- Residual connections (Transformer 2017) -> Hyper-Connections -> mHC -> mHC-lite
- Weighted sum of permutation matrices = doubly stochastic matrix = Birkhoff polytope
- Mass-preserving soft shuffling of feature channels (no norm explosion)
- Applied to multi-omics: each omics block encoded independently, then soft-mixed

Phase 1 (Pan-cancer, 32 classes) — COMPLETE:
- Real TCGA data loaded: 8,314 patients, 9,844 features (3217 mRNA + 383 miRNA + 3139 Methy + 3105 CNV)
- Both models ~97.7% accuracy — task too easy, mixer never engaged (alpha stays uniform)
- Pipeline fully validated end-to-end on real data

Phase 2 (BRCA PAM50 subtypes, 5 classes) — IN PROGRESS:
- Harder task where cross-omics routing is biologically necessary
- Data: GS-BRCA from CMOB Figshare, ~1000 patients
- Expected: non-uniform alpha, structured cross-omics flow heatmap

Environment:
- MacBook Pro M1, conda env "jlab", PyTorch 2.10.0, MPS=True
- Repo cloned at ~/Cancer-Multi-Omics-Benchmark/
- Real data at: ~/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/Pan-cancer/Original/
- Notebooks in ~/CMOB/ (or local CMOB/ folder in JupyterLab)
