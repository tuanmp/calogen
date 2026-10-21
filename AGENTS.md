# AGENTS.md — calogen

Model-agnostic generative calorimetry. Self-contained project at
`/global/cfs/cdirs/m3443/usr/pmtuan/calogen` (moved out of `caloinn_lightning`).

## Environment

- **Always use `uv`.** Python >= 3.10.
- `HDF5_USE_FILE_LOCKING=FALSE` is **mandatory** for anything touching HDF5.
- The project has its own venv: `cd calogen && uv sync`.

## Entrypoints

| Purpose | Command (from `calogen/`) |
|---|---|
| **Train** | `HDF5_USE_FILE_LOCKING=FALSE uv run python main.py fit --config configs/caloinn/train_lemurs_fcceeallegro.yaml` |
| **Test** | `HDF5_USE_FILE_LOCKING=FALSE uv run pytest tests/ -v` |
| **Single test** | `HDF5_USE_FILE_LOCKING=FALSE uv run pytest tests/test_icaloflow_model.py -v` |
| **Evaluate (LEMURS)** | `HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/evaluate_lemurs.py --config configs/eval/lemurs_base.yaml` |

`main.py` = LightningCLI entrypoint, run from the repo root:
`uv run python main.py ...`. The `calogen` package is installed editable by
`uv sync` (see Layout), so no `sys.path` manipulation is needed.

## Layout

The importable package `calogen` is an **alias for the `src/` directory**
(`package-dir = { "calogen" = "src" }` in `pyproject.toml`); the repo root holds
only the entrypoint, configs, tests and reference material.

```
calogen/                # repo root
  pyproject.toml
  main.py               LightningCLI entrypoint
  src/                  -> imported as `calogen` (calogen == src)
    __init__.py
    core/               model-agnostic: data_util, xml_handler, sharded/lemurs
                        data, postprocess, trainer, callbacks/, plotting,
                        diagnostics
    models/
    base.py           GenerativeModel protocol + BaseGenerativeModule
      registry.py       name -> (LightningModule, DataModule) plugin lookup
      caloinn/          cinn, distributions, vblinear, blocks/, lightning,
                        datamodule, classifier.py, features/ (convert, torch_hlf)
      icaloflow/        model, lightning, datamodule, classifier.py
    classifiers/        model-agnostic NCE base (base.py) + calibration.py
    eval/               LEMURS truth-vs-generated evaluation: io, observables,
                        metrics, plots, report (library + scripts CLI)
    mcsampling/         (Phase 3) acceptance-rejection Monte-Carlo resampling
    compat.py           legacy module-name aliases for old checkpoints
  configs/              gen-model + NCE classifier configs, binning XMLs
  assets/icaloflow_q0/  flow1-mean.pt / flow1-cov.pt (NCE multivariate-normal base)
  tests/                transferred/adapted tests + phase parity tests
  reference/            parity capture scripts + artifacts
```

Imports are always `from calogen.<subpkg> import ...` (e.g.
`from calogen.core import data_util`), resolved from `src/`.

## Phase status

- **Phase 1:** generative models CaloINN + iCaloFlow, training and generation.
- **Phase 2 (this):** NCE-style classifiers (CaloINN + iCaloFlow) and score calibration.
- **Phase 3:** Monte-Carlo (acceptance-rejection) resampling (not yet transferred).

### Phase 2 classifiers

- `calogen.classifiers.base` — `MLP`, `NCEClassifier` (generic Lightning loop),
  `NoiseContrastiveDensityRatioEstimator` (latent NCE math). Imports **no** model.
- `calogen.models.caloinn.classifier` — `CaloINNCLF` (shower-space HF classifier)
  and `CaloINNLatentNCE` (latent NCE; the old
  `classifier.NoiseContrastiveDensityRatioEstimator`).
- `calogen.models.icaloflow.classifier` — `ICaloFlowNCE` + MVN / MVNFlow1 /
  MVNFlow1V2 / MVNFlow1V3 / MVNConditionAgnostic(Flow1/Flow1V2) subclasses.
- `calogen.models.caloinn.features` — shower→classifier-input conversion
  (`convert.py`, `torch_hlf.py`), reusing `core.xml_handler` and
  `core.caloch_eval.HighLevelFeatures` instead of the old `_xml_handler`/`_hlf` duplicates.

## Key gotchas

1. `torch.set_default_dtype(torch.float32)` is set in `main.py` before imports.
2. No double-noise: `training_step` does not add noise; the dataloader does.
3. Sharded backend supports `num_workers > 0`. `use_sharded: false` (streaming)
   is **not** available in Phase 1 and raises `NotImplementedError`.
4. `_SkipLastTwoScheduler` wraps the LR scheduler (legacy OneCycleLR behavior).
5. iCaloFlow flows are float64; latents/shower outputs are returned float32.
6. Configs keep absolute `/pscratch` data and checkpoint paths (data is shared,
   not copied into this repo).
7. `class_path` values must be fully qualified (`calogen.core.callbacks.*`,
   `calogen.models.caloinn.*`, ...).
8. Do **not** add the repo's parent directory to `sys.path`: it contains the
   `calogen/` repo dir (which has no `__init__.py`), creating a PEP 420
   namespace package that shadows the installed `calogen` and breaks
   `calogen.core` imports. Rely on the editable install (`uv sync`).
9. **Classifier RNG order is intentional.** In each classifier `__init__` the
   generator is built *before* the MLP (into a local, then `super().__init__`),
   because `MLP`'s eager `nn.Linear` layers consume RNG. Reordering changes
   random init and breaks parity with the original runs.
10. `--trainer.limit_val_batches 1` is parsed as float `1.0` = 100% of val
    (Lightning treats floats in [0,1] as fractions). Use a value like `0.01`
    for a partial val pass.
11. Old generator checkpoints embed `_class_path: lightning_module.CaloINNLightningModule`.
    `import calogen.compat` registers legacy module aliases; `main.py` does this
    automatically. Embedded `xml_path` in those checkpoints is relative, so
    configs pass `kw_overrides.xml_path`.
12. NCE configs reference large generator checkpoints by absolute path (not
    copied). q0 assets live in `assets/icaloflow_q0/`.
13. **LEMURS within-layer cell order is r-major.** `core.lemurs._transpose_and_flatten`
    maps raw `(R,Phi,Z)` to layer-major `(Z,R,Phi)` with `index = r*N_phi + phi`.
    `XMLHandler`'s per-layer eta/phi arrays are **phi-major**, so coordinate-dependent
    observables (center-of-energy, widths) must use the LEMURS-native geometry in
    `calogen.eval.observables`, not `HighLevelFeatures`' XML coordinates. Order-independent
    aggregates (E_tot, layer energies, active fraction) are safe either way.
14. `calogen.eval` evaluation uses `threshold_mev` (e.g. `0.015`) to match the generator
    noise cut; pass it via `configs/eval/lemurs_base.yaml`. `configs/` is gitignored
    (environment-local).

## Parity

Phase-1 verification against `caloinn_lightning` (read-only). All checks passed.

1. **Unit tests** (`uv run pytest tests/ -q`): 80 passed, 17 skipped.
   The 4 failures + 1 error are **pre-existing** in the old repo (confirmed by
   running the same tests against `caloinn_lightning`): stale
   `test_icaloflow_data` batch-key assertions and a `load_from_checkpoint`
   hyperparameter-parsing error in `test_torch_postprocess`.
2. **Weights/output equality** (`tests/test_phase1_parity.py`, reference from
   `reference/capture_phase1.py`): identical state-dict structure and matching
   `log_prob` / `sample` / `inverse` / student / helper outputs for CaloINN
   (float32, rtol 1e-5) and iCaloFlow (float64, rtol 1e-9).
3. **Short parallel training** (login-node GPUs, same seed, old vs new):
   - iCaloFlow Flow-1: identical per-step losses and `val_loss=934.0`,
     `train_loss_epoch=962.0`.
   - CaloINN LEMURS: identical `val_loss=18208.6484375`,
     `val_loss_per_dim=2.790597677230835`, identical checkpoint filename.

Parity artifacts: `reference/phase1_ref.pt`; run logs under
`/pscratch/sd/p/pmtuan/calogen_parity/`.

## Phase 2 parity

Verified against `caloinn_lightning` (read-only). All checks passed.

1. **Unit tests:** suite total 117 passed / 17 skipped; the same 4 pre-existing
   failures + 1 error as Phase 1. New: `tests/test_classifier.py` (MLP, shared
   NCE math, conversion) and `tests/test_calibration.py`.
2. **Feature-conversion equality** (`tests/test_phase2_parity.py`): new
   `calogen.models.caloinn.features.convert` equals the original `mcmc.convert`
   on fixed inputs (rtol/atol 1e-6).
3. **Short parallel NCE training** (login-node GPUs, same seed, old vs new):
   - iCaloFlow Flow1V3: identical `train_loss` (0.68251693, 0.67107463,
     0.64999509) and identical `val_loss=0.6215556`, `val_aucroc=0.8293750`,
     `val_brier=0.2147228`, `val_ece=0.1932890`.
   - CaloINN latent NCE: identical `train_loss` (0.69521308, 0.70496917,
     0.74462593), identical `val_loss=0.5571153`, `val_aucroc=0.9890341`,
     `val_brier=0.1838809`; `val_ece` matches to ~4e-8.

Parity configs/logs: `/pscratch/sd/p/pmtuan/calogen_parity/`.
