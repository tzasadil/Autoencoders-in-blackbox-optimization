---
name: Transform Optimization
overview: Fit a simple target-specific rotation and translation for each of 1k training functions so their values match the currently observed black-box DOE as closely as possible.
todos:
  - id: define-loss
    content: Use one simple fitting loss that matches the current DOE normalization.
    status: pending
  - id: shortlist-strategy
    content: Reduce the training library to 1k functions and give each one a small transform-fitting budget.
    status: pending
  - id: parametrize-transform
    content: Choose a plain bounded translation plus angle-based rotation parameterization for dimensions 2, 5, and 10.
    status: pending
  - id: prototype-solver
    content: Prototype one constrained low-budget solver path and compare VAE selection against direct fitted-loss selection.
    status: pending
isProject: false
---

# Transform Optimization Analysis

The current DOE2Vec pipeline in [doe2vec/doe2vec.py](doe2vec/doe2vec.py) evaluates a large library of generated expression strings on selected normalized sample points, normalizes each resulting value row, trains the VAE on those rows, encodes the target DOE row, and selects the nearest latent function. During optimization, this surrogate is retrained repeatedly from the currently observed samples in [evo.py](evo.py), where `surrogate.train(true_xs, true_ys, opt=optimizer)` is called after accepted CMA batches.

The new requirement changes the comparison rule between candidate functions. A candidate should no longer be judged in its raw form. Instead, each training function should first be allowed a target-specific local alignment through rotation and translation, and only then be compared to the currently observed black-box DOE. In other words, the method should compare each function in the best fitted version that can be obtained under a common bounded optimization budget.

## Methodological Position

The intended methodology is therefore not a transformless shortlist followed by expensive fitting of a few survivors. That would conflict with the stated goal, because functions would still be filtered before being given the chance to align to the target. A more coherent design is to reduce the library itself to 1k training functions and to give every one of those functions the same low-budget local transform optimization. The resulting fitted function rows are then used as the actual training data for DOE2Vec.

This preserves the VAE as the main focus of the thesis. The proposed extension is not a replacement for DOE2Vec, but a target-specific preprocessing layer placed before latent encoding. The primary selector therefore remains the VAE operating on fitted rows. A direct fitted-loss selector can still be added as a comparison baseline, but it should be presented as an auxiliary benchmark rather than as the main method.

## Transform Model

The input points should remain in the same normalized coordinate system already used by the DOE code:

`u_i = clip((x_i + 5) / 10, eps, 1 - eps)`

For a generated function `f_j`, the fitted input is defined by

`z_i = c + t + R @ (u_i - c)`

where `c` denotes the current local center, `t` is a bounded translation vector, and `R` is a rotation matrix. The center should be taken from the current local search state, preferably the CMA mean or the mean of the selected local training points, so that the transformation acts relative to the part of the search space currently being modeled rather than relative to the global origin.

The transformed candidate row is then

`g_i = f_j(z_i)`

and the quality of this fitted candidate is measured using a single normalized loss,

`loss = MSE(zscore(y), zscore(g))`

where `y` is the currently observed target DOE row. This keeps the fitting objective aligned with the current row-normalized DOE representation and focuses the comparison on local response shape rather than absolute scale.

## Parameterization

The transform family should be described as a bounded finite-dimensional optimization problem. Translation should be optimized directly as a box-constrained variable, for example `t in [-tau, tau]^d`. Rotation should be parameterized through sequential Givens rotations. This yields a simple and standard coordinate description of orthogonal transformations without introducing heavier manifold machinery.

Under this parameterization, the optimization variables are:

- `d` bounded translation parameters
- `d(d-1)/2` bounded rotation-angle parameters

Thus, the transform search space remains explicit, finite-dimensional, and bounded in every tested dimension. In 2D this gives one rotation angle and two box-constrained translation parameters, in 5D it gives 10 rotation angles and five translation parameters, and in 10D it gives 45 rotation angles and 10 translation parameters.

Transformed points may be clipped back to `[eps, 1 - eps]^d` in the first implementation. That is acceptable as a practical boundary handling rule for the prototype. If needed later, the clipping can be replaced by a smooth penalty without changing the overall methodological framing.

## Optimization Protocol

Each candidate function should solve the same constrained local fitting problem under the same computational budget. This equal-budget rule is important, because it gives a uniform comparison protocol across the 1k-function library. The method does not need to claim globally optimal alignment; it only needs to define a consistent target-specific fitting procedure that is applied fairly to every candidate.

For the first prototype, a single derivative-free optimizer is sufficient. A practical choice is `scipy.optimize.minimize` with Powell's method, initialized at the identity rotation and zero translation. The optimization budget can be intentionally small, since precision is less important here than methodological consistency and computational feasibility inside the repeated surrogate retraining loop.

This produces a clear thesis-level statement: each candidate function is first locally aligned to the currently observed DOE by solving the same bounded transform optimization problem, and DOE2Vec is then trained on the aligned candidate rows rather than on the raw library.

## Integration With DOE2Vec

After fitting all 1k candidate functions, the pipeline should evaluate only the fitted versions of those functions and use the resulting rows to construct the DOE training matrix. The VAE is then trained on that fitted matrix exactly as before, except that the latent space now represents aligned candidate functions rather than raw candidate functions.

The primary selection rule should remain latent-distance selection in this fitted latent space. This keeps the central thesis claim intact: the VAE is still the main representation and matching mechanism. At the same time, the pipeline should also record which function would have been chosen by direct fitted loss alone, so that the contribution of the latent representation can be assessed experimentally.

## Scope And Cost

The full 250k-function library is not appropriate for this extension, because even a cheap local fit repeated that many times would be too expensive inside the optimization loop. The proposed regime is therefore to cap the training library at 1k functions. This is small enough to make per-function fitting realistic, while still leaving the VAE with a nontrivial function population to compress and organize.

The important computational choice is not to make the fit precise, but to keep it uniform. Every candidate receives the same low-budget transform search, after which comparison proceeds on the fitted outputs. This is a stronger and more defensible methodology than trying a small set of hand-picked transforms, because it formulates the step as a genuine constrained optimization problem rather than as a small heuristic menu.

## Baselines And Validation

The experimental section should compare at least the following variants:

- no transform baseline
- translation-only fitting
- rotation-only fitting
- rotation-plus-translation fitting
- VAE selection on fitted rows
- direct fitted-loss selection on fitted rows

Evaluation should report both surrogate Spearman correlation on candidate populations and final BBOB optimization performance. This is necessary because better local DOE fit does not automatically imply better downstream optimization behavior.

## JAX Rewrite Note

A JAX rewrite is optional and not required for the first prototype. The current library is generated as Python expression strings, so a proper JAX rewrite would require changing the internal function representation rather than merely swapping one numerical backend for another. For the initial implementation, the simpler and more proportionate choice is to keep the existing function representation and use a small-budget SciPy optimizer for the per-function alignment step.
