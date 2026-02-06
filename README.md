## openinfer-oinf

Python tooling for the `.oinf` model format. Use this package to encode
dataclass-based models, verify binaries, and inspect sizevars and tensors.

### Install deps
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r ../requirements.txt
```

### Generate a model
```bash
python examples/mlp_regression_oinf.py
python verify_oinf.py res/models/mlp_regression.oinf
```

### Tests
```bash
python tests/run_oinf_tests.py
```

Docs: https://github.com/arsalan-awnari/openinfer/tree/main/docs/sphinx/modules/openinfer-oinf
