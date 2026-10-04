# 🏥 Upgraded KG-CMI: Knowledge Graph Enhanced Cross-Mamba Interaction for Medical Visual Question Answering

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **College Minor Project**: Enhanced **KG-CMI** framework for Medical Visual Question Answering (Med-VQA) featuring **Relational Graph Convolutional Networks (R-GCN)**, **Question-aware Cross-Mamba Interaction**, and **Multi-Task Learning**.

---

## 📌 Project Overview

Medical Visual Question Answering (Med-VQA) assists clinical decision-making by enabling intelligent systems to answer complex natural language queries about medical images (Radiology, CT, MRI, X-ray). 

Standard Med-VQA models struggle with domain-specific reasoning and fine-grained anatomical localization. This project upgrades the **KG-CMI** architecture with:
1. **Relational Graph Convolutional Network (R-GCN)** for multi-hop organ-disease knowledge graph reasoning.
2. **Cross-Modal Mamba (CMM)** selective scanning for linear complexity vision-language interaction.
3. **Free-Form Answer Enhanced Multi-Task Learning (FAMT)** to improve open-ended clinical query response generation.

---

## 🏗️ Architecture & Modules

```
                        ┌──────────────────────────────┐
                        │   Input Medical Image & Text │
                        └──────────────┬───────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
  ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
  │ Vision Encoder   │       │ Text Encoder     │       │ Knowledge Graph  │
  │ (ViT / BioMed)   │       │ (PubMedBERT)     │       │ (SLAKE / UMLS)   │
  └─────────┬────────┘       └─────────┬────────┘       └─────────┬────────┘
            │                          │                          │
            └────────────────────┬─────┴──────────────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ 2-Layer R-GCN KGE     │ (Relational Graph Reasoning)
                     └───────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │  Cross-Mamba (CMM)    │ (Selective Scan Fusion)
                     └───────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │  Multi-Task Head      │ (Classification + T5 Aux)
                     └───────────────────────┘
```

### Key Technical Contributions & Novelties:
- **`models/rgcn_kge_module.py`**: Upgrades 1-layer homogeneous GAT to a 2-layer R-GCN with basis decomposition ($B=4$) to differentiate relation types (`is_located_in`, `manifests`, `adjacent_to`).
- **`KG_CMI_Colab_Training.ipynb`**: End-to-end training notebook pre-configured for Google Colab GPUs (T4 / L4).
- **Domain-Specific Encoders**: Extensible support for PubMedBERT and BioMedCLIP backbones.

---

## 📊 Benchmark Datasets

Evaluated across 3 standard Med-VQA benchmarks:
* **VQA-RAD**: Radiological images across 3 organ systems (1,793 total training QA pairs divided into 1,613 train / 180 validation pairs, and 451 official test benchmark QA pairs from Hugging Face `flaviagiammarino/vqa-rad`).
* **SLAKE**: Multi-modal English medical VQA dataset (450 train / 96 val / 96 test images).
* **OVQA**: Large-scale orthopedic medical visual question-answering dataset.

---

## 🚀 Quick Start & Google Colab Training

### Option A: Run on Google Colab (Recommended)
1. Upload `KG_CMI_Colab_Training.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Execute all cells to automatically install `mamba-ssm`, download datasets, and train the model!

### Option B: Local Setup & Dependencies

```bash
# Clone this repository
git clone https://github.com/Anjali-2807/KG-CMI-MedVQA.git
cd KG-CMI-MedVQA

# Install PyTorch & Vision-Language Dependencies
pip install -r requirements.txt
pip install causal-conv1d>=1.2.0 mamba-ssm torch-geometric
```

### Train Model
```bash
python main.py --config ./lavis/projects/blip2/train/vqa_rad.yaml
```

---

## 📂 Repository Structure

```text
KG-CMI-MedVQA/
├── models/
│   └── rgcn_kge_module.py          # Upgraded Relational GCN KGE PyTorch Module
├── KG_CMI_Colab_Training.ipynb     # Google Colab GPU Training Notebook
├── push_to_github.sh                # Helper script for GitHub sync
├── main.py                         # Main training & evaluation loop
├── lavis/                          # Vision-Language backbones & configs
├── m3ae/                           # Cross-Mamba interaction modules
└── README.md                       # Project documentation
```

---

## 🎓 College Minor Project Details
* **Author**: Anjali Tiwari
* **Project Domain**: Artificial Intelligence / Medical Image Processing / Deep Learning
* **Framework**: PyTorch, HuggingFace Transformers, Mamba-SSM, PyTorch Geometric
