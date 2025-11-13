
# 📘 Educational Response Prediction Using Machine Learning

This project explores multiple machine learning approaches for predicting whether a student will correctly answer a question, given a sparse student–question response matrix. The work compares classical collaborative filtering, statistical models, deep learning methods, and an ensemble, highlighting how different model assumptions affect performance and generalization.

## 🔍 Project Overview
Educational datasets are typically sparse and high-dimensional, making prediction challenging.  
This project evaluates four different modelling strategies:

- k-Nearest Neighbors (user-based and item-based)
- Item Response Theory (1-parameter logistic model)
- Neural Network Autoencoder (baseline + enhanced architecture)
- Weighted Ensemble of all three models

The goal is to determine which methods produce the most accurate predictions and why.

## 🧠 Models Implemented

### 1. k-Nearest Neighbors (KNN)
- Implemented both user-based and item-based collaborative filtering.  
- Used `KNNImputer` with NaN-Euclidean distance.  
- Tuned k across multiple values to optimize validation accuracy.  
- Achieved up to **0.6897–0.6916 validation accuracy**.

### 2. Item Response Theory (IRT)
- Implemented the **1PL (Rasch) model** for predicting correctness probability.  
- Derived gradients of the log-likelihood with respect to student ability (θ) and question difficulty (β).  
- Trained using gradient ascent and analyzed item difficulty through ICC plots.  
- Achieved **0.7034 test accuracy**.

### 3. Neural Network Autoencoder
- Built a shallow baseline autoencoder and an enhanced deep denoising autoencoder using PyTorch.  
- Added batch normalization, ReLU activations, mini-batch training, Adam optimizer, learning rate scheduling, and early stopping.  
- Performed ablation studies to compare the effect of each enhancement.  
- Baseline outperformed the enhanced model due to dataset sparsity and over-regularization.

### 4. Weighted Ensemble
- Combined predictions from KNN, IRT, and NN models using a weighted average.  
- Weights were proportional to validation accuracy.  
- Final ensemble achieved **0.7206 test accuracy**, outperforming all individual models.

## 📊 Key Results

| Model | Test Accuracy |
|-------|--------------|
| KNN | 0.6847 |
| IRT | 0.7034 |
| Neural Network | 0.6915 |
| **Ensemble** | **0.7206** |

The ensemble generalized best across unseen data.

## 🧪 Techniques & Skills Demonstrated
- Data preprocessing for sparse matrices  
- Collaborative filtering (user- and item-based)  
- Statistical modelling with IRT  
- PyTorch neural network development  
- Gradient derivations & optimization  
- Hyperparameter tuning  
- Ensemble learning  
- Model evaluation and visualization  

## 📂 Dataset
- 542 students  
- 1774 questions  
- Values: 1 = correct, 0 = incorrect, NaN = unanswered  
- Highly sparse matrix typical in educational platforms

## 🚀 Future Work
- Explore 2PL/3PL IRT models  
- Integrate question metadata (topics, difficulty tiers)  
- Investigate transformer-based student modeling  
- Use improved denoising strategies for NN training  
- Compare against matrix factorization methods  

