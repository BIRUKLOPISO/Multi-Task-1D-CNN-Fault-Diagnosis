# Multi-Task 1D-CNN for Fault Detection, Classification and Section Identification

This repository contains the implementation of the paper:

**Multi-Task 1D-CNN for Fault Detection, Classification and Section Identification in Active Distribution Networks with Inverter-Based Resources**

## Overview

We propose a multi-task one-dimensional Convolutional Neural Network (1D-CNN) that simultaneously performs:

- Fault type classification (11 classes)
- Faulted section identification (12 sections)

The model works **end-to-end** on raw three-phase voltage and current waveforms.

## Results

| Task                            | Accuracy | Macro F1-score |
|---------------------------------|----------|----------------|
| Fault Type Classification       | 91.81%   | 91.15%         |
| Faulted Section Identification  | 71.64%   | 71.97%         |

## Repository Structure
├── model/
│   └── model.py
├── data/
│   └── generate_dataset.py
├── train/
│   └── train.py
├── evaluate/
│   └── evaluate.py
├── requirements.txt
└── README.md

## Installation

```bash
pip install -r requirements.txt
Usage
1. Generate Dataset
python data/generate_dataset.py
2. Train the Model
python train/train.py
3. Evaluate
python evaluate/evaluate.py
Citation
If you use this code, please cite the paper (link will be added after publication).
License
This project is licensed under the MIT License.
