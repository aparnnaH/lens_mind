# LensMind Installation

LensMind requires Python 3.11 or newer.

Install the project dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

OpenCLIP defaults are stored in `pyproject.toml` under
`tool.lensmind.embeddings`.

The intended embedding device policy is:

- use `mps` when `torch.backends.mps.is_available()`
- otherwise use `cpu`

Embedding generation is not implemented yet. Installation and application startup
must not download or initialize OpenCLIP model weights.
