

<h2 align="center">
  [ICLR 2026] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion
</h2>

<div align="center">
  <a href="https://fcbfcb1998.github.io/">Chen-Bin Feng</a><sup>1,2*</sup>,&nbsp;&nbsp;
  Youyang Sha<sup>1*</sup>,&nbsp;&nbsp;
  <a href="https://capsule2077.github.io/">Longfei Liu</a><sup>1</sup>,&nbsp;&nbsp;
  Yongjun Yu<sup>1</sup>,&nbsp;&nbsp;
  <a href="https://www.fst.um.edu.mo/personal/cmvong/">Chi Man Vong</a><sup>2†</sup>,&nbsp;&nbsp;
  <a href="https://xuanlong-yu.github.io/">Xuanlong Yu</a><sup>1†</sup>,&nbsp;&nbsp;
  <a href="https://xishen0220.github.io">Xi Shen</a><sup>1†</sup>
</div>

<p align="center">
  <i>
  1. <a href="https://intellindust-ai-lab.github.io">Intellindust AI Lab</a> &nbsp;&nbsp;
  2. University of Macau <br>
  * Equal Contribution &nbsp;&nbsp; † Corresponding Author
  </i>
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2604.11998">
    <img alt="3rd Place" src="https://img.shields.io/badge/3rd_Place-CVPR_NTIRE2026_CDFSOD-yellow">
  </a>
  <a href="https://github.com/Intellindust-AI-Lab/FSOD-VFM/blob/master/LICENSE">
    <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
  </a>
  <a href="https://arxiv.org/abs/2602.03137">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.03137-red">
  </a>
  <a href="https://intellindust-ai-lab.github.io/projects/FSOD-VFM/">
    <img alt="project webpage" src="https://img.shields.io/badge/Webpage-FSODVFM-purple">
  </a>
  <a href="https://github.com/Intellindust-AI-Lab/FSOD-VFM/issues">
    <img alt="issues" src="https://img.shields.io/github/issues/Intellindust-AI-Lab/FSOD-VFM?color=olive">
  </a>
  <a href="https://github.com/Intellindust-AI-Lab/FSOD-VFM">
    <img alt="stars" src="https://img.shields.io/github/stars/Intellindust-AI-Lab/FSOD-VFM">
  </a>
  <a href="mailto:shenxi@intellindust.com">
    <img alt="Contact Us" src="https://img.shields.io/badge/Contact-Email-yellow">
  </a>
</p>

---

<p align="left">
  <strong>FSOD-VFM</strong> is a framework for <em>few-shot object detection</em> leveraging powerful <strong>vision foundation models</strong> (VFMs).  
  It integrates three key components:
  <br><br>
  🔹 <strong>Universal Proposal Network (UPN)</strong> for category-agnostic bounding box generation<br>
  🔹 <strong>SAM2</strong> for accurate mask extraction<br>
  🔹 <strong>DINOv2</strong> features for efficient adaptation to novel object categories<br><br>
  To address over-fragmentation in proposals, FSOD-VFM introduces a novel <strong>graph-based confidence reweighting</strong> strategy for refining detections.
</p>




<p align="center">
  <strong>If you find our work useful, please give us a ⭐!</strong>
</p>

---

<p align="center">
  <img src="./teaser/overv.png" alt="Overview" width="98%">
</p>

---

## 🚀 Updates 
- [x] **\[2026.4.20\]** Based on our solution FSOD-VFM, **NJUST-KMG** won the 3rd place in the [CVPR NTIRE 2026 CDFSOD Challenge](https://arxiv.org/abs/2604.11998).
- [x] **\[2026.3.2\]** Add Radiov4 model.
- [x] **\[2026.2.3\]** Initial release of FSOD-VFM.

---

## 🧭 Table of Contents
1. [Datasets](#1-datasets)  
2. [Quick Start](#2-quick-start)  
3. [Usage](#3-usage)  
4. [Citation](#4-citation)  
5. [Acknowledgement](#5-acknowledgement)  


---

## 1. Datasets
Put all datasets under `FSOD-VFM/dataset/`:

```bash
git clone https://github.com/Intellindust-AI-Lab/FSOD-VFM
cd FSOD-VFM 
mkdir dataset
```

### Pascal VOC
Download Pascal VOC from [http://host.robots.ox.ac.uk/pascal/VOC](http://host.robots.ox.ac.uk/pascal/VOC),  
then put it under `/dataset/` following structure:

```shell
    dataset/PascalVOC/
    ├── VOC2007/
    ├── VOC2007Test/
    │   └── VOC2007
    │   │  ├── JPEGImages
    │   │  └── ...
    │   └── ...
    └── VOC2012/
````

### COCO

Download COCO from [https://cocodataset.org](https://cocodataset.org) and organize it as:

```shell
dataset/coco/
├── annotations/
├── train2017/
├── val2017/
└── test2017/
```

### CD-FSOD

Download CD-FSOD from [https://yuqianfu.com/CDFSOD-benchmark/](https://yuqianfu.com/CDFSOD-benchmark/),
and organize as:

```shell
dataset/CDFSOD/
    ├── ArTaxOr/...
    ├── clipart1k/...
    ├── DIOR/...
    ├── FISH/...
    ├── NEU-DET/...
    └── UODD/...
```

---

## 2. Quick Start

This `dev` branch’s [`fsod.yml`](fsod.yml) installs **PyTorch 2.7.1+cu128**.
UPN ops (`chatrex/upn/ops`) are a CUDA extension
([MultiScaleDeformableAttention](https://github.com/fundamentalvision/Deformable-DETR),
vendored from [ChatRex](https://github.com/IDEA-Research/ChatRex)).
`nvcc` must match PyTorch’s CUDA (**12.8**), not a newer system toolkit (e.g. `/usr/local/cuda` 13.0).

### Environment Setup

```bash
conda env create -f fsod.yml
conda activate FSODVFM
```

Install a **CUDA 12.8** compiler and headers *inside the env* (do **not** install
`gcc=9.5.0` / `gxx=9.5.0` from conda-forge — that conflicts with `cuda-nvcc` 12.8).
System `g++` 13.x is fine.

```bash
conda activate FSODVFM
conda install -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-libraries-dev=12.8 ninja -y
```

Point the shell at the conda toolkit and PyTorch’s `lib` (needed for **build and later runs**):

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
```

Check before compiling:

```bash
which nvcc          # must be $CONDA_PREFIX/bin/nvcc, not /usr/local/cuda/bin/nvcc
nvcc --version      # 12.8
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # 2.7.1+cu128  12.8
```

Persist the library path so `libc10.so` is found after the ops install:

```bash
conda env config vars set LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
conda deactivate && conda activate FSODVFM
```

`CUDA_HOME` / `PATH` still need setting in each new shell (or add them after `conda activate`).

### DINOv2 Installation
```bash
# Ensure the operation is performed inside the /FSOD-VFM directory
git clone https://github.com/facebookresearch/dinov2.git
```

### UPN Installation

With `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` set as above:

```bash
cd chatrex/upn/ops
pip install -v --no-build-isolation -e .
python -c "import MultiScaleDeformableAttention as MSDA; print('ok', MSDA)"
```

If `cusparse.h` is missing, `cuda-libraries-dev=12.8` was not installed.
If import fails with `libc10.so: cannot open shared object file`, `LD_LIBRARY_PATH` does not include `site-packages/torch/lib`.

Upstream README used `conda-forge gcc=9.5.0 gxx=9.5.0` then `pip install -v -e .`.
That solver pin is incompatible with this env’s CUDA 12.8 stack; skip it.

### SAM2 Installation

```bash
# Ensure the operation is performed outside the /FSOD-VFM directory
cd ../../../../
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

### Checkpoints

```bash
# Make sure the checkpoints folder is inside the project root (FSODVFM/checkpoints). 
cd FSOD-VFM && mkdir checkpoints && cd checkpoints 
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth
```

* For RADIOv4, download "c-radio_v4-h_half.pth.tar" from [NVlabs/RADIO](https://github.com/NVlabs/RADIO/tree/main) to /model_cache/torch_hub/checkpoints.
---

## 3. Usage

### Pascal VOC

**Using DINOv2**

```bash
sh run_scripts/run_pascal.sh
```

**Using RADIOv4**

```bash
sh run_scripts/run_pascal_radio.sh
```

**Tips:**

* Modify `--json_path` for different splits (`split1`, `split2`, `split3`) and shot settings (`1shot`, `5shot`, etc.).
* Modify `--target categories` for different splits.
* Adjust hyperparameters:

  * `--min_threshold`: UPN confidence threshold (default: `0.01`)
  * `--alp`: alpha for graph diffusion
  * `--lamb`: decay parameter for graph diffusion
* To fix shell script issues:

  ```bash
  sed -i 's/\r$//' run_scripts/run_pascal.sh
  ```

<p align="center">
  <img src="./teaser/exp1.png" alt="Overview" width="98%">
</p>

---

### COCO

**Using DINOv2**

```bash
sh run_scripts/run_coco.sh
```

**Using RADIOv4**

```bash
sh run_scripts/run_coco_radio.sh
```

**Tips:**

* Modify `--json_path` for `10shot` or `30shot`.
* Target categories are fixed to the standard COCO 20 classes.

<p align="center">
  <img src="./teaser/exp2.png" alt="Overview" width="98%">
</p>

---

### CD-FSOD

**Using DINOv2**

```bash
sh run_scripts/run_cdfsod.sh
```

**Using RADIOv4**

```bash
sh run_scripts/run_cdfsod_radio.sh
```

**Tips:**

* Modify `--json_path`, `--test_json`, and `--test_img_dir` for different subsets (e.g., `ArTaxOr`, `DIOR`).
* For `DIOR`, use:

  ```
  --test_img_dir ./dataset/CDFSOD/DIOR/test/new_test/
  ```

<p align="center">
  <img src="./teaser/fsod2.png" alt="Overview" width="98%">
</p>

---

## 4. Citation

If you use **FSOD-VFM** in your research, please cite:

```latex
@inproceedings{feng2025fsodvfm,
  title={FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion},
  author={Feng, Chen-Bin and Sha, Youyang and Liu, Longfei and Yu, Yongjun and Vong, Chi Man and Yu, Xuanlong and Shen, Xi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

---

## 5. Acknowledgement

Our work builds upon excellent open-source projects including
[No-Time-To-Train](https://github.com/miquel-espinosa/no-time-to-train),
[SAM2](https://github.com/facebookresearch/sam2/tree/main),
[ChatRex](https://github.com/IDEA-Research/ChatRex),
[DINOv2](https://github.com/facebookresearch/dinov2), and
[RADIOv4](https://github.com/NVlabs/RADIO/tree/main).
We sincerely thank their authors for their contributions to the community.



