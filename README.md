# Credit Card Fraud Detection Project

## 1. Project Overview

This project demonstrates a complete machine learning workflow to build a model capable of identifying fraudulent credit card transactions. The script uses a real-world dataset, preprocesses the data, handles the significant class imbalance inherent in fraud detection problems, trains a robust classification model, and evaluates its performance.

## 2. Dataset

The project utilizes the "Credit Card Fraud Detection" dataset available on Kaggle.

* **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

* **Description:** The dataset contains transactions made by European cardholders in September 2013. It consists of 284,807 transactions, of which only 492 (0.172%) are fraudulent.

* **Features:** The features are mostly anonymized (`V1` to `V28`) as a result of a Principal Component Analysis (PCA) transformation. The only features that have not been transformed are `Time` and `Amount`. The target variable is `Class`, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.

## 3. Methodology

The Python script (`credit_fraud_detection.py`) follows these key steps:

1. **Data Loading & Exploration:** The script loads the dataset and performs an initial analysis to understand its structure, summary statistics, and check for missing values.

2. **Data Preprocessing:** The `Time` and `Amount` columns are scaled using `StandardScaler` to ensure all features have a similar scale, which is crucial for many machine learning algorithms.

3. **Handling Class Imbalance:** Due to the highly imbalanced nature of the data, a **Random Undersampling** technique is applied. This creates a new, balanced dataset by randomly removing samples from the majority class (non-fraudulent transactions) to match the number of fraudulent transactions. This prevents the model from being biased towards the majority class.

4. **Model Training:** A **Random Forest Classifier** is trained on the balanced dataset. This model is chosen for its robustness and ability to handle complex, non-linear relationships in the data.

5. **Model Evaluation:** The trained model is evaluated on a test set using several key metrics:

   * **Accuracy, Precision, Recall, and F1-Score:** To measure the model's effectiveness.

   * **Classification Report:** A detailed breakdown of the model's performance.

   * **Confusion Matrix:** A visual representation of the model's true vs. predicted classifications.

6. **Feature Importance:** The script identifies and visualizes the most important features that contribute to predicting fraud, providing insights into the key drivers of fraudulent activity.

## 4. Requirements

To run this project, you need Python 3 and the following libraries:

* pandas

* numpy

* scikit-learn

* matplotlib

* seaborn

You can install these dependencies using pip:

```
pip install pandas numpy scikit-learn matplotlib seaborn

```

## 5. How to Run

1. Ensure all the required libraries are installed.

2. Save the code as a Python file (e.g., `credit_fraud_detection.py`).

3. Run the script from your terminal:

   ```
   python credit_fraud_detection.py
   
   ```

The script will execute all the steps, print the evaluation results to the console, and display visualizations for class distribution, the confusion matrix, and feature importance.

## 6. Results Summary

The model achieves high performance in identifying fraudulent transactions. The evaluation metrics (Precision, Recall, F1-Score) from the script's output demonstrate its effectiveness. The feature importance analysis reveals that certain PCA-transformed features (like `V14`, `V12`, `V10`) are strong indicators of fraud.
