## Customer Churn Prediction (Machine Learning Project)

This project predicts whether a telecom customer is likely to **churn (leave the company)** using machine learning models.

The goal is to help businesses identify customers at risk of leaving so that they can take preventive action.

---

## Dataset

Telco Customer Churn Dataset

Features include:

- tenure
- MonthlyCharges
- TotalCharges
- Contract type
- Payment method
- Internet services
- Customer demographics

Target variable:
- 1 = Customer leaves
- 0 = Customer stays

---

## Machine Learning Models Used

1. Logistic Regression  
2. Random Forest Classifier  
3. XGBoost Classifier  

Models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Evaluation Metrics

The following metrics were used:

- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score

---

## Model Performance

| Model | Accuracy | ROC-AUC |
|------|------|------|
| Logistic Regression | ~0.80 | ~0.84 |
| Random Forest | ~0.79 | ~0.82 |
| XGBoost | ~0.80 | ~0.84 |

The **best model selected automatically** was:XG BOOST(GRID)
---

## Model Explainability

We used **SHAP (SHapley Additive exPlanations)** to understand model predictions.

Key factors affecting churn:

- Contract type
- Monthly charges
- Tenure
- Internet service type
- Payment method

---

## Project Structure
customer_churn_ml/
│
├── FINAL_CHURN_ML_CODE.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_model_features.pkl
├── roc_curve_logistic.png
├── shap_summary.png
├── requirements.txt
└── README.md

---

## Key Learnings

This project demonstrates:

- Data preprocessing
- Feature encoding
- Train-test splitting
- Model training
- Cross-validation
- Hyperparameter tuning
- Model comparison
- Model explainability using SHAP

---

## Author

Nitesh Nankani

Machine Learning & AI Systems Learner
