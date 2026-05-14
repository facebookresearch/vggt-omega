# VGGT-Omega

VGGT-Omega is a feed-forward reconstruction model for static and dynamic scenes.
Given one or more input views, the release is intended to provide pretrained
camera and depth inference, visualization tools, and export utilities for
downstream 3D workflows.[^release]

This repository is in the initial release-skeleton stage. Model code,
checkpoint links, demos, and export utilities will be added in follow-up
commits.

## Release Scope

This repository will include:

- pretrained camera and depth inference
- lightweight installation as a Python package
- visualization and export utilities for inference outputs
- documentation and assets needed for the public release

This repository will not include:

- training or fine-tuning code
- paper benchmark evaluation scripts
- scripts to reproduce the full training data pipeline

## Installation

Install PyTorch for your CUDA environment, then install the package in editable
mode:

```bash
git clone <repo-url>
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```

## Status

The first implementation milestone is a minimal inference API that loads a
pretrained VGGT-Omega checkpoint and predicts cameras and depth maps from an
image sequence. The public API, checkpoint names, and demo commands will be
documented here once they are available.

## License

VGGT-Omega is released under the FAIR Noncommercial Research License. See
[LICENSE](LICENSE) for the full license text.

## Citation

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-\Omega},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Schonberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

[^release]: This Release is intended to support the open source research community.
