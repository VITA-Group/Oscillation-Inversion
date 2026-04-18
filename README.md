# Oscillation Inversion

### Training-Free Image and Video Enhancement Through Oscillated Latents in Large Flow Models

**AAAI 2026 Oral**

[Yan Zheng](https://yanyanzheng96.github.io)<sup>1</sup>,
Zhenxiao Liang<sup>1</sup>,
Xiaoyan Cong<sup>2</sup>,
Yi Yang<sup>3</sup>,
Lanqing Guo<sup>1</sup>,
Yuehao Wang<sup>1</sup>,
Peihao Wang<sup>1</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>1</sup>

<sup>1</sup>University of Texas at Austin, <sup>2</sup>Brown University, <sup>3</sup>The University of Edinburgh

[[Project Page]](https://yanyanzheng96.github.io/oscillation_inversion/) [[Paper]](docs/data/teaser.pdf)

## Abstract

We explore the oscillatory behavior observed in inversion methods applied to large-scale flow models, including text-to-image and text-to-video. By employing an augmented fixed-point-inspired iterative approach to invert real-world images, we observe that the solution does not achieve convergence, instead oscillating between distinct clusters. Through both experiments on synthetic data, text-to-image and text-to-video, we demonstrate that these oscillating clusters exhibit notable semantic coherence. We offer theoretical insights, showing that this behavior arises from oscillatory dynamics in flow models. Building on this understanding, we introduce a simple and fast distribution transfer technique that facilitates training-free image and video editing/enhancement. Furthermore, we provide quantitative results demonstrating the effectiveness of our method on tasks such as image enhancement, editing, and reconstruction. Notably, our approach enables the transformation of image-only enhancers and editors into lightweight, video-capable tools—without additional training—highlighting its practical versatility and impact.

## Method Overview

<p align="center">
  <img src="docs/data/teaser.png" width="90%">
</p>

**Key idea:** Fixed-point iteration in flow models causes oscillation between semantic clusters rather than convergence. We exploit this behavior through *Group Inversion* — simultaneously inverting a group of images to push outputs toward the high-quality data manifold.

**Core algorithm (Oscillation Inversion):**
```
z^{(k+1)}_{t_0} = y - (sigma_0 - sigma_{t_0}) * v_theta(z^{(k)}_{t_0}, sigma_{t_0})
```

**Group Inversion:**
```
z^{(k+1)}_{t_0} = y_{(k mod m)} - (sigma_0 - sigma_{t_0}) * v_theta(z^{(k)}_{t_0}, sigma_{t_0})
```

## Repository Structure

```
Oscillation-Inversion/
├── src/                              # Core implementation
│   ├── flux_utils.py                 # Oscillation Inversion with FLUX (single target)
│   └── flux_utils_multi.py           # Group Inversion (multi-target)
│
├── diffusers_local/                  # Modified diffusers pipelines
│   ├── pipelines/flux/               # Custom FLUX pipeline
│   └── models/transformers/          # Custom transformer modules
│
├── scripts/
│   ├── run_oscillation_inversion.py  # Single-image oscillation inversion
│   ├── run_group_inversion.py        # Group inversion with multiple targets
│   ├── run_depth_align.py            # Depth-aligned inversion
│   ├── image_enhancement/            # Image enhancement experiments (Sec. 6.1)
│   │   ├── run_blur.py               # Deblurring
│   │   ├── run_noise.py              # Denoising
│   │   ├── run_downsample.py         # Super-resolution (4x)
│   │   ├── run_compress.py           # Compression artifact removal
│   │   ├── batch_*.sh                # Batch processing scripts
│   │   └── metric_*.py               # PSNR/LPIPS/FID evaluation
│   └── video_enhancement/            # Video enhancement experiments (Sec. 6.2)
│       ├── run_video.py              # Video inversion
│       ├── run_video_makeup.py       # Video makeup transfer
│       └── batch_*.sh                # Batch processing scripts
│
├── notebooks/
│   ├── theory_toy_example.ipynb      # Toy Gaussian theory visualization (Sec. 5)
│   ├── oscillation_analysis.ipynb    # Fixed-point oscillation analysis
│   └── image_editing_demo.ipynb      # Image editing/recoloring demo
│
├── demo/                             # Demo images
│   ├── glassgirl.png                 # Sample input image
│   ├── women/                        # Face enhancement demo
│   ├── makeup/                       # Makeup transfer demo
│   └── texture/                      # Texture synthesis demo
│
├── configs/                          # Configuration files
│   ├── config.py                     # Configuration dataclass
│   └── depth_align.yaml              # Depth alignment config
│
└── docs/                             # Project webpage
    ├── index.html
    └── data/                         # GIFs, PDFs for webpage
```

## Installation

```bash
git clone https://github.com/VITA-Group/Oscillation-Inversion.git
cd Oscillation-Inversion
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- NVIDIA GPU with >= 24GB VRAM (A6000 recommended)

## Quick Start

### 1. Oscillation Inversion (Single Image)

```bash
python scripts/run_oscillation_inversion.py
```

This runs fixed-point iteration on a source-target image pair using FLUX.1-schnell, demonstrating the oscillation phenomenon between semantic clusters.

### 2. Group Inversion (Multi-Target Enhancement)

```bash
python scripts/run_group_inversion.py
```

This runs the augmented group inversion with multiple target images, enabling distribution transfer for image enhancement.

### 3. Image Enhancement on CelebA (Section 6.1)

```bash
# Process blurred CelebA images
cd scripts/image_enhancement
bash batch_blur.sh

# Compute metrics (PSNR, LPIPS)
python metric_blur.py
```

Available degradation types: `blur`, `noise`, `downsample`, `compress`

### 4. Video Enhancement (Section 6.2)

```bash
cd scripts/video_enhancement
bash batch_video.sh
```

### 5. Theory Visualization (Section 5)

Open `notebooks/theory_toy_example.ipynb` in Jupyter to reproduce the toy Gaussian mixture experiment demonstrating oscillation dynamics in rectified flow.

## Models

This codebase uses the following pretrained models (automatically downloaded from HuggingFace):

| Model | Usage |
|-------|-------|
| [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Primary T2I model (4-step distilled) |
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Alternative T2I model |
| [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) | T2V model for video enhancement |

## Results

### Image Enhancement (CelebA, Table 1)

| Method | Denoise PSNR | Denoise LPIPS | Deblur PSNR | Deblur LPIPS | 4xSR PSNR | 4xSR LPIPS |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| BlindDPS | - | - | 23.56 | 0.257 | 21.82 | 0.345 |
| Piscart | 28.21 | 0.15 | 30.23 | 0.15 | 29.68 | 0.12 |
| **Ours** | 25.50 | **0.13** | 26.90 | **0.12** | 25.44 | 0.17 |

### Video Enhancement (VFHQ, Table 2)

| Method | T-LPIPS | CLIP_TSC |
|--------|:---:|:---:|
| Baseline | 0.0324 | 0.9823 |
| **Ours** | **0.0285** | **0.9847** |

## Citation

```bibtex
@inproceedings{zheng2026oscillation,
  title={Oscillation Inversion: Training-Free Image and Video Enhancement Through Oscillated Latents in Large Flow Models},
  author={Zheng, Yan and Liang, Zhenxiao and Cong, Xiaoyan and Yang, Yi and Guo, Lanqing and Wang, Yuehao and Wang, Peihao and Wang, Zhangyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Acknowledgements

This project builds upon [FLUX](https://github.com/black-forest-labs/flux), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), and [diffusers](https://github.com/huggingface/diffusers). We thank the authors for their excellent work.
