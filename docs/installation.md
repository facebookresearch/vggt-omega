# Installation

VGGT-Omega is tested as a CUDA inference package. Install PyTorch for your
machine first, then install this repository.

## PyTorch

Choose the PyTorch wheel that matches your CUDA driver. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

The package does not pin a specific CUDA wheel in `pyproject.toml` because
PyTorch installation is environment-specific.

## Package

```bash
git clone <repo-url>
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```

## Notes

- A CUDA GPU is expected for inference.
- No separate DINOv3 checkpoint download is required.
- The released model checkpoints are raw PyTorch `state_dict` files.
- VGGT-Omega checkpoint access requires review; after approval, keep the
  checkpoint as a local file and pass that local path to scripts and demos.
- The code does not use `PyTorchModelHubMixin`; checkpoints are loaded
  explicitly with `torch.load` and `model.load_state_dict`.
