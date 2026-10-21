# calogen

Model-agnostic generative calorimetry models: CaloINN and iCaloFlow.

- **Phase 1:** generative models + training/generation.
- **Phase 2:** NCE-style classifiers.
- **Phase 3:** Monte-Carlo (acceptance-rejection) resampling.

## Quick start

```bash
cd calogen
uv sync
# generative model
HDF5_USE_FILE_LOCKING=FALSE uv run python main.py fit --config configs/caloinn/train_lemurs_fcceeallegro.yaml
# NCE classifier (Phase 2)
HDF5_USE_FILE_LOCKING=FALSE uv run python main.py fit --config configs/icaloflow/nce_icaloflowMVNFlow1V3_M1_L256.yaml
```

See `AGENTS.md` for details.
