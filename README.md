# Dimensionality Reduction and Visualization

## Project Overview

This project explores the effect of dimensionality reduction techniques on classification performance using the Fashion-MNIST dataset. The goal is to reduce high-dimensional image data to lower dimensions while maintaining class discriminability and improving computational efficiency.

We evaluate three popular techniques:

- **PCA** (Principal Component Analysis)
- **LDA** (Linear Discriminant Analysis)
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding)

Each technique is evaluated based on:
- Visual interpretability of the transformed data
- Classification performance (using a Gaussian classifier)
- Sensitivity to subspace dimensionality

---

## Dataset Description

- **Fashion-MNIST** consists of 10,000 grayscale images (28×28) from 10 fashion categories.
- Each image is flattened into a 784-dimensional vector.
- The dataset is split **class-wise** into:
  - 5000 training samples
  - 5000 testing samples

---

## Implemented Methods

### 1. Principal Component Analysis (PCA)

- Data is centered and projected onto orthogonal bases (eigenvectors of the covariance matrix).
- The top 20 eigenvectors are visualized as 28×28 grayscale images.
- Gaussian classification is performed in subspaces of varying dimension (1 to 350, log-spaced).
- Results show:
  - Low-dimensional projections underfit
  - Very high dimensions can overfit
  - Best performance occurs around **50–100 components**

### 2. Linear Discriminant Analysis (LDA)

- LDA is a supervised technique that maximizes between-class separation.
- At most C - 1 = 9 components are extracted (for 10 classes).
- LDA basis vectors are reshaped and visualized.
- Gaussian classification in LDA space shows performance converges quickly — most gain is from the first few dimensions.

### 3. t-SNE Visualization

- t-SNE is used for 2D visualization of class structure.
- Preprocessing: Standardization using StandardScaler
- Parameters used:
  - n_components=2
  - perplexity=30
  - init='pca'
  - max_iter=1000
- Result: Clear visual clusters, with some overlap between similar classes (e.g., shirt vs. pullover)

---

## Results Summary

| Method | Best Error (Test) | Notes |
|--------|-------------------|-------|
| PCA + Gaussian Classifier | ~20.1% | Optimal around 60–100 dimensions |
| LDA + Gaussian Classifier | ~20.1% | Only 9 dimensions used |
| t-SNE (visualization only) | — | Not used for classification |

Both PCA and LDA reach similar classification accuracy, with LDA using fewer dimensions thanks to supervised optimization.

---

## Project Structure

```text
.
├── data/                   # Fashion-MNIST data files (.txt, .npz)
├── img/                    # Generated plots (PCA, LDA, t-SNE)
├── src/                    # Python scripts for each method
│   ├── preprocess.py
│   ├── pca_analysis.py
│   ├── lda_analysis.py
│   └── tsne_visualization.py
├── Report.pdf
└── README.md              
