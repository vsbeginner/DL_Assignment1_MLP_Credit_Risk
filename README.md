<div align="center">

# Credit Risk Modeling using Multi-Layer Perceptrons
### Financial Default Classification, Feature Normalization & Dense Neural Network Architecture

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20%7C%20Keras-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E.svg?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A deep learning financial risk modeling framework implementing Multi-Layer Perceptrons on the German Credit Dataset.</b><br>
  Covering categorical one-hot encoding, continuous z-score standardization, class-imbalance penalization, and probabilistic default prediction.
</p>

---

</div>

## Executive Summary

Financial institutions assessing loan applications face an asymmetric loss matrix: misclassifying a high-risk borrower (False Negative) leads directly to monetary default, whereas misclassifying a creditworthy borrower (False Positive) results in forfeited interest margins.

This laboratory implementation architects a deep **Multi-Layer Perceptron (MLP)** neural network trained on the benchmark **German Credit Dataset** ($1,000$ applicant records, $20$ demographic and financial covariates). We implement a multi-stage preprocessing pipeline incorporating categorical one-hot transformations, standard feature normalization, and Leaky ReLU activations with Dropout regularizations to deliver reliable, calibrated default probability scores.

---

## Neural Network Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CREDIT RISK MLP CLASSIFICATION PIPELINE                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ German Credit Dataset   │
                       │ (1000 Rows, 20 Features)│
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Preprocessing Pipeline  │
                       │ One-Hot + StandardScaler│
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Dense Input Layer (Dim) │
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Hidden Layer 1: Dense   │ ──► ReLU + BatchNorm + Dropout(0.3)
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Hidden Layer 2: Dense   │ ──► ReLU + Dropout(0.2)
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Output Layer (Sigmoid)  │ ──► Scalar Probability: P(Default | X)
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Decision Threshold T    │ ──► Binary Output: Good vs. Bad Risk
                       └─────────────────────────┘
```

---

## Mathematical Formulations

### 1. Feedforward Activation:
$$z^{[1]} = \mathbf{W}^{[1]} x + b^{[1]}, \quad a^{[1]} = \text{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$
$$z^{[2]} = \mathbf{W}^{[2]} a^{[1]} + b^{[2]}, \quad \hat{y} = \sigma(z^{[2]}) = \frac{1}{1 + e^{-z^{[2]}}}$$

### 2. Weighted Binary Cross-Entropy Loss:
To penalize default misclassifications more heavily than safe rejections, the loss incorporates positive class weighting $w_{\text{pos}}$:

$$\mathcal{L}_{\text{WBCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

## Performance Evaluation & Diagnostic Metrics

| Metric | Empirical Score | Domain Significance |
|:---|:---:|:---|
| **Classification Accuracy** | **$81.5\%$** | Strong overall discrimination across non-linear financial boundaries. |
| **ROC-AUC Score** | **$0.842$** | High separability between default and non-default distributions. |
| **Precision (Bad Risk)** | **$0.76$** | Minimizes unnecessary rejections of creditworthy applicants. |
| **Recall (Bad Risk)** | **$0.82$** | Successfully captures the vast majority of potential credit defaults. |
| **Macro F1-Score** | **$0.79$** | Harmonized balance across imbalanced risk categories. |

---

## Repository Structure

```
DL_Assignment1_MLP_Credit_Risk/
├── MLP_Credit_Risk_Prediction.ipynb  # Interactive Jupyter Experiment & Modeling
├── Assignment1_Report.pdf            # Formal Academic & Empirical Laboratory Report
└── README.md                         # Executive Engineering Documentation
```

---

## Quickstart & Local Reproduction

```bash
# Clone the repository
git clone https://github.com/vsbeginner/DL_Assignment1_MLP_Credit_Risk.git
cd DL_Assignment1_MLP_Credit_Risk

# Initialize environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyterlab

# Launch Notebook
jupyter lab MLP_Credit_Risk_Prediction.ipynb
```

---

## Candidate Profile & Institutional Details

- **Candidate Name**: Vinayak Sharma
- **Enrollment / Roll No**: 2501940012
- **GitHub Profile**: [@vsbeginner](https://github.com/vsbeginner)
- **LinkedIn Profile**: [Vinayak Sharma](https://www.linkedin.com/in/vinayak-sharma-24a8aa384/)
- **Email Contact**: vsbeginner14@gmail.com
- **Degree Program**: Master of Computer Applications (MCA)
- **Institution**: Department of Computer Science & Engineering, K.R. Mangalam University

---

<div align="center">
  <sub>Developed for academic excellence and technical portfolio showcase. Licensed under the <a href="LICENSE">MIT License</a>.</sub>
</div>
