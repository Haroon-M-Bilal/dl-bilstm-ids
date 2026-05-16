# DL-BiLSTM-IDS: Robust and Privacy-Preserving IoT Intrusion Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ART 1.16](https://img.shields.io/badge/ART-1.16-blueviolet.svg)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
[![Flower 1.6](https://img.shields.io/badge/Flower-1.6-3a8fb7.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-000000.svg)](https://peps.python.org/pep-0008/)

> An extension of the **DL-BiLSTM** IoT intrusion detection framework (Wang et al., 2023) addressing three open gaps: **class imbalance**, **adversarial robustness**, and **privacy-preserving federated training**. Evaluated on **CICIoT2023** under both 8-class and 34-class taxonomies.

---

## 📑 Abstract

Deep learning–based intrusion detection systems for the Internet of Things (IoT) have shown strong performance on benchmark datasets, but three practical concerns remain underexplored: (i) severe **class imbalance** that biases models toward majority traffic classes, (ii) brittleness against **adversarial perturbations** that can evade detection, and (iii) the **privacy risk** of centralized training over distributed IoT traffic.

This repository extends the DL-BiLSTM architecture with: (1) class-weighted cross-entropy loss with SMOTE-style minority-class emphasis, (2) adversarial hardening via FGSM and PGD training using IBM's Adversarial Robustness Toolbox (ART), and (3) FedAvg-based federated learning simulation across three independent IoT client nodes. Experiments are run on the CICIoT2023 dataset under two configurations: a fine-grained **34-class** taxonomy and a coarser **8-class** taxonomy aligned with Wang et al. (2023).

---

## 🧱 Repository Layout

```
dl-bilstm-ids/
├── train_pipeline.py        # Full end-to-end pipeline (single script)
├── requirements.txt         # Pinned dependencies (PyTorch 2.1 + CUDA 12.1)
├── LICENSE
├── README.md
└── results_*/               # Per-run outputs (created by pipeline)
    ├── baseline_model.pth
    ├── smote_model.pth
    ├── hardened_model.pth
    ├── federated_model.pth
    ├── cm_baseline.png
    ├── cm_smote.png
    ├── cm_hardened.png
    ├── cm_federated.png
    ├── federated_convergence.png
    └── results.json
```

The pipeline is organized as a single `train_pipeline.py` script with six modes (`preprocess`, `baseline`, `smote`, `hardened`, `federated`, `ablation`, `all`).

---

## ⚙️ Installation

### Prerequisites

- Python **3.11**
- NVIDIA GPU with **CUDA 12.1** (tested on RTX 4080 Laptop, 12 GB VRAM)
- **Windows 10/11** or Linux
- ~50 GB free disk space for the full CICIoT2023 dataset

### Setup

```bash
# Clone the repository
git clone https://github.com/Haroon-M-Bilal/dl-bilstm-ids.git
cd dl-bilstm-ids

# Create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

# Install PyTorch with CUDA 12.1 first
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
            --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 4080 ...
```

### Dataset

Download **CICIoT2023** from the Canadian Institute for Cybersecurity and unzip the CSV files into any folder (e.g. `D:\datasets\cic_iot_2023\` on Windows or `~/datasets/cic_iot_2023/` on Linux).

- Source: [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

> The dataset itself is **not redistributed** in this repository.

---

## 🚀 Usage

All commands are run from the repository root. Random seed is fixed (`seed=42`) for reproducibility.

### Full end-to-end run (34-class)

```bash
python train_pipeline.py --data_dir D:\datasets\cic_iot_2023 --mode all
```

### Full end-to-end run (8-class, for direct comparison with Wang et al. 2023)

```bash
python train_pipeline.py --data_dir D:\datasets\cic_iot_2023 --mode all --collapse_classes 8
```

### Individual stages

```bash
python train_pipeline.py --data_dir <path> --mode baseline
python train_pipeline.py --data_dir <path> --mode smote
python train_pipeline.py --data_dir <path> --mode hardened
python train_pipeline.py --data_dir <path> --mode federated
python train_pipeline.py --data_dir <path> --mode ablation
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | (required) | Path to CICIoT2023 CSV folder |
| `--mode` | `all` | One of `preprocess`, `baseline`, `smote`, `hardened`, `federated`, `ablation`, `all` |
| `--collapse_classes` | `None` | `8` for Wang-style 8-class grouping, `2` for binary, or omit for full 34-class |
| `--epochs` | `30` | Training epochs per phase |
| `--batch_size` | `512` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate (AdamW + cosine schedule) |

Outputs (trained `.pth` weights, confusion matrices, federated convergence plot, `results.json`) are saved into a per-run folder under `results_<n_classes>class/`.

---

## 📊 Results on CICIoT2023

### 8-Class Taxonomy (Wang et al. comparison)

| Configuration       | Accuracy   | Precision  | Recall     | F1         |
|---------------------|------------|------------|------------|------------|
| Baseline DL-BiLSTM  | **0.8072** | 0.8181     | 0.8072     | **0.7965** |
| + SMOTE-Weighted    | 0.7738     | 0.8190     | 0.7738     | 0.7877     |
| + Adv. Hardened     | 0.7529     | 0.8026     | 0.7529     | 0.7670     |
| Federated (N=3)     | **0.8084** | 0.8300     | 0.8084     | **0.7941** |
| Wang et al. (2023)  | 0.9313     | —          | —          | —          |

> The Wang et al. comparison is interpreted with care: they evaluate on a restricted **Recon + Mirai** subset of CICIoT2023, while this work uses the full 8-class taxonomy including BruteForce, Web, Spoofing, and DoS/DDoS variants — categories that introduce substantially greater inter-class confusion (notably between Recon and Spoofing).
>
> Notably, the federated model **slightly exceeds** the centralized baseline (80.84 % vs 80.72 % accuracy), confirming that decentralized training imposes no measurable accuracy penalty in this setting.

### 34-Class Fine-Grained Taxonomy

| Configuration       | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Baseline DL-BiLSTM  | 0.7260   | 0.7311    | 0.7260 | 0.7108 |
| + SMOTE-Weighted    | 0.7082   | 0.7214    | 0.7082 | 0.6964 |
| + Adv. Hardened     | 0.6431   | 0.6590    | 0.6431 | 0.6090 |
| Federated (N=3)     | 0.7125   | 0.7304    | 0.7125 | 0.6917 |

### Adversarial Robustness (8-Class, ε = 0.1)

| Attack | Baseline Accuracy | Baseline ASR | Hardened Accuracy | Hardened ASR |
|--------|-------------------|--------------|-------------------|--------------|
| Clean  | 0.8072            | 0.1928       | 0.7529            | 0.2471       |
| FGSM   | 0.2446            | 0.7554       | **0.7496**        | **0.2504**   |
| PGD    | 0.1092            | 0.8908       | **0.3925**        | **0.6075**   |

> Without hardening, the model collapses under FGSM (24 % accuracy) and PGD (11 %). Adversarial training restores FGSM accuracy to **74.96 %** — a 50-point gain — and PGD accuracy from 11 % to **39.25 %**, at the cost of a modest ~5 pp drop in clean accuracy.
>
> *ASR = Attack Success Rate.*

### Federated Convergence (8-Class, 3 clients, 10 rounds)

| Round | Avg. Client Loss | Val. Accuracy | Val. F1 |
|-------|------------------|---------------|---------|
| 1     | 0.5177           | 0.7783        | 0.7522  |
| 5     | 0.4665           | 0.8041        | 0.7877  |
| 10    | **0.4513**       | **0.8089**    | **0.7946** |

Validation loss decreased monotonically from **0.518 → 0.451** with smooth, oscillation-free convergence — confirming that FedAvg aggregation is numerically stable for this problem.

### Federated Convergence (34-Class, 3 clients, 10 rounds)

| Round | Avg. Client Loss | Val. Accuracy | Val. F1 |
|-------|------------------|---------------|---------|
| 1     | 0.8454           | 0.6427        | 0.5923  |
| 5     | 0.7679           | 0.7020        | 0.6765  |
| 10    | **0.7422**       | **0.7108**    | **0.6902** |

---

## 🧠 Methodology Summary

1. **Preprocessing.** 63 CICIoT2023 CSVs are loaded with per-class capping (50,000 max/class for 34-class, 100,000 for 8-class), Min-Max normalized, and reduced to 40 components via Incremental PCA.
2. **Architecture.** A 2-stage DNN (256 → 128) feeds a 2-layer bidirectional LSTM (hidden=128), followed by a LayerNorm + Dropout classifier head.
3. **Class imbalance.** Cross-entropy with class weights inversely proportional to class frequency.
4. **Adversarial training.** Each minibatch is augmented with FGSM and PGD perturbations (ε=0.1). The model is held in `train()` mode during attack generation to preserve gradient flow through the BiLSTM's recurrent layers — switching to `eval()` mode breaks CUDNN backward passes on LSTMs.
5. **Federated learning.** FedAvg across 3 clients, 10 communication rounds, 2 local epochs per round, IID shards.

---

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{dlbilstmids2026,
  author       = {Haroon Muhammad Bilal},
  title        = {{DL-BiLSTM-IDS}: Robust and Privacy-Preserving IoT Intrusion Detection},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Haroon-M-Bilal/dl-bilstm-ids}}
}
```

This work extends:

```bibtex
@article{wang2023dlbilstm,
  title   = {A Lightweight {IoT} Intrusion Detection Model Based on {DL-BiLSTM}},
  author  = {Wang and others},
  journal = {arXiv preprint},
  year    = {2023}
}
```

---

## 🤝 Acknowledgments

- Wang et al. (2023) for the original DL-BiLSTM architecture.
- The Canadian Institute for Cybersecurity (CIC) for the CICIoT2023 dataset.
- IBM's [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox).
- The [Flower](https://flower.ai/) federated learning framework.

---

## 📄 License

This project is released under the [MIT License](LICENSE).
