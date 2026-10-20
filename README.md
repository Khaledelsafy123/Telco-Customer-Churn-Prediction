# Telco Customer Churn Prediction

A machine learning project that predicts customer churn in the telecommunications industry by training and comparing five classification models.

[View Notebook on Kaggle](https://www.kaggle.com/code/khalod05/telco-churn-prediction-ml-model-comparison)

## Overview
Customer churn occurs when customers stop using a company's services. Predicting churn can help businesses identify customers who may leave and make better retention decisions.

This project covers data preprocessing, feature encoding, feature scaling, hyperparameter tuning, and model evaluation.

## Objective

- Build classification models to predict customer churn.
- Optimize model hyperparameters using `RandomizedSearchCV`.
- Compare models using Accuracy, F1 Score, and Recall.
- Select the final model based on cross-validation accuracy.

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Original columns:** 21
- **Target variable:** `Churn` (`Yes` / `No`)

The dataset contains customer demographic information, subscribed services, contract details, tenure, and billing information.

## Data Preprocessing

1. Removed `customerID` because it is an identifier rather than a predictive feature.
2. Converted `TotalCharges` to numeric values after removing whitespace and replacing empty strings with `0`.
3. Split the data into training and testing sets using an 80/20 split, `random_state=42`, and stratification.
4. Encoded the target labels using `LabelEncoder`.
5. Applied One-Hot Encoding to categorical features.
6. Standardized numerical features using `StandardScaler`.

The resulting feature matrices contain **45 features**, with 5,634 training samples and 1,409 testing samples.

## Models

Five classification models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Linear SVM
- RBF SVM
- Polynomial SVM (`degree=2`)

## Hyperparameter Tuning

`RandomizedSearchCV` was used to search for suitable hyperparameters for each model.

- **Cross-validation:** 3 folds
- **Iterations:** 10 parameter combinations per model
- **Scoring metric:** Accuracy
- **Random state:** `42`
- **Parallel processing:** `n_jobs=-1`

The model with the highest mean cross-validation accuracy was selected.

## Results

All metrics are percentages.

| Model               | CV Accuracy | Test Accuracy |    Test F1 | Test Recall |
| ------------------- | ----------: | ------------: | ---------: | ----------: |
| Logistic Regression |      80.55% |        79.91% |     56.93% |      50.00% |
| KNN                 |      78.70% |        77.22% |     56.33% |  **55.35%** |
| Linear SVM          |      80.56% |    **80.13%** | **59.18%** |      54.28% |
| RBF SVM             |  **80.76%** |        79.42% |     55.38% |      48.13% |
| Polynomial SVM      |      80.65% |        79.42% |     56.19% |      49.73% |

## Final Model

**Selected Model: RBF SVM**

RBF SVM achieved the highest cross-validation accuracy of **80.76%**, so it was selected as the final model. Its test accuracy was **79.42%**.

Linear SVM achieved the highest test accuracy (**80.13%**) and F1 Score (**59.18%**), while KNN achieved the highest test Recall (**55.35%**). Therefore, RBF SVM was selected based on cross-validation accuracy, but it did not perform best across every evaluation metric.

## Technologies

- Python
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Kaggle

## How to Run

1. Clone or download this repository.

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. Open `Telco_Customer_Churn_Prediction.ipynb` in Jupyter Notebook.

5. Ensure the dataset path matches the notebook, then run the cells in order.

---

*Created as a machine learning project to practice classification, model comparison, and hyperparameter tuning.*

## Author

Khaled Elsafy
[GitHub profile](https://github.com/Khaledelsafy123)
