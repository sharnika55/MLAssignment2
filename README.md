# ML Assignment 2 – Classification Models

## Problem Statement
The objective of this assignment is to design, implement, evaluate, and compare multiple supervised machine learning classification models on a medical dataset, and deploy the trained models using Streamlit for interactive evaluation.

## Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)

- **Number of Instances:** 569  
- **Number of Features:** 30 numerical features  
- **Target Variable:** `diagnosis`  
  - M → Malignant (1)  
  - B → Benign (0)

The dataset is used to predict whether a breast tumor is malignant or benign based on cell nucleus characteristics.

## Models Implemented
The following classification models were implemented, trained, and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

All models were implemented using Python. Training and evaluation were performed using **scikit-learn**, and **XGBoost** was implemented using the `xgboost` library. Trained models were saved as `.pkl` files for reuse and deployment.

## Model Comparison Results

| Model               | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|--------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression | 0.9737 | 0.9974 | 0.9762 | 0.9535 | 0.9647 | 0.9439 |
| Decision Tree       | 0.9386 | 0.9369 | 0.9091 | 0.9302 | 0.9195 | 0.8701 |
| KNN                 | 0.9474 | 0.9815 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| Naive Bayes         | 0.9649 | 0.9974 | 0.9756 | 0.9302 | 0.9524 | 0.9253 |
| Random Forest       | 0.9649 | 0.9949 | 0.9756 | 0.9302 | 0.9524 | 0.9253 |
| XGBoost             | 0.9561 | 0.9951 | 0.9524 | 0.9302 | 0.9412 | 0.9064 |

## Observations

| Model | Observation |
|------|-------------|
| Logistic Regression | High accuracy, stable, and highly interpretable |
| Decision Tree | Fast but prone to overfitting |
| KNN | Performance sensitive to feature scaling |
| Naive Bayes | Simple and efficient baseline model |
| Random Forest | Strong ensemble model with good generalization |
| XGBoost | Best overall balance of accuracy and AUC |

## Deployment
The trained models were deployed using **Streamlit Community Cloud**, allowing users to upload test data, select a classification model, and view evaluation metrics along with confusion matrices.

## Conclusion
This project demonstrates a complete machine learning workflow including data preprocessing, model training, performance evaluation using multiple metrics, and deployment through a web-based interface. Ensemble models such as Random Forest and XGBoost achieved strong predictive performance, while Logistic Regression offered excellent interpretability with competitive accuracy.
