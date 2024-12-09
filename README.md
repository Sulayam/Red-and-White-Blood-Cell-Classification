# Blood Cell Images Based Classification

## Overview
This project focuses on classifying blood cell images into four types: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. Automated classification can enhance medical diagnostics by reducing manual effort and improving accuracy.

## Problem Statement
The goal is to build a machine learning model that can accurately categorize blood cell images based on their morphological and staining characteristics.

## Pipeline Workflow
1. **Data Preparation**:
   - Image extraction from ZIP file
   - DataFrame creation from training and testing CSVs
   - Image segmentation using color masks and contour detection

2. **Feature Engineering**:
   - Local Binary Pattern (LBP) for texture feature extraction
   - Principal Component Analysis (PCA) for dimensionality reduction
   - Data standardization

3. **Model Training and Evaluation**:
   - Stacking classifier with KNN, SVM, and XGBoost as base models
   - Hyperparameter tuning using GridSearchCV
   - F-Score (Beta, Micro) for model performance evaluation

4. **Predictions and Submission**:
   - Predictions on test dataset
   - Submission file generation

## Segmented Images
![Segmented Images](segmented_images.png)

## Results
- Best model parameters determined through GridSearchCV
- Submission file `submission_2_sulay.csv` created with predicted categories

## Future Enhancements
1. Data augmentation to address class imbalance
2. Explore deep learning techniques for feature extraction and classification
3. Bayesian optimization for faster and better hyperparameter tuning
4. Implement explainability tools like SHAP
