# AEGIS: Explainable Customer Churn Risk Model

## Overview

Most churn prediction models stop at answering one question:

**"Will this customer leave?"**

The AEGIS Churn Risk Model goes a step further by answering:

- Why is the customer likely to churn?
- Which business factors contribute the most?
- How confident is the prediction?
- Which customers should be prioritized for retention?

Instead of functioning as a black-box classifier, AEGIS transforms customer behavior into transparent business risk signals, making predictions that are easier to understand and act upon.

---

## Problem Statement

Organizations often struggle with customer churn because traditional prediction models provide limited business insight. While they may predict churn accurately, they rarely explain *why* a customer is at risk.

The objective of this project is to develop an interpretable churn prediction framework that combines machine learning with business-driven risk scoring to support informed retention strategies.

---

## AEGIS Framework

The model introduces three business-oriented risk indicators:

- **URS (Usage Risk Score)** – Measures customer engagement and service usage.
- **BRS (Billing Risk Score)** – Captures financial and payment-related behavior.
- **SRS (Service Risk Score)** – Evaluates service quality and customer relationship indicators.

These signals are transformed, standardized, and combined into a unified churn risk score that remains both explainable and statistically reliable.

---

## Project Workflow

The notebook follows an end-to-end machine learning pipeline:

1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Construction of URS, BRS, and SRS
5. Feature scaling and normalization
6. Model training
7. Five-fold cross-validation
8. Threshold optimization
9. Risk segmentation
10. Business recommendations

---

## Key Features

- Explainable churn prediction framework
- Business-driven feature engineering
- Interpretable risk scoring
- Probability-based customer ranking
- Risk band classification
- Cross-validation for reliable performance
- Actionable retention recommendations

---

## Model Evaluation

The model is evaluated using multiple classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-Score
- Cross-validation Performance

Special emphasis is placed on **Recall**, ensuring that potential churners are identified effectively for proactive intervention.

---

## Business Insights

The model not only predicts churn but also helps answer practical business questions such as:

- Which customers require immediate attention?
- Which behavioral factors contribute most to churn?
- How should retention campaigns be prioritized?
- Which customer segments present the highest business risk?

These insights make the model useful for customer success teams, marketing departments, and business analysts.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Repository Structure

```
AEGIS-Churn-Risk-Model/
│
├── AEGIS_Churn_Risk_Model.ipynb
├── dataset.csv
├── README.md
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/tdivyasri/AEGIS-Telecom-Churn-Risk-Model
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `AEGIS_Churn_Risk_Model.ipynb` and execute the notebook cells in sequence.

---

## What I Learned

This project strengthened my understanding of:

- Feature engineering for customer analytics
- Explainable machine learning
- Business-oriented risk modeling
- Cross-validation and model evaluation
- Customer segmentation
- Translating predictive models into business decisions

---

## Future Improvements

Possible extensions include:

- SHAP-based feature explanations
- Interactive Streamlit dashboard
- Real-time churn prediction API
- Automated model retraining pipeline
- Integration with CRM platforms

---

## License

This project is intended for educational, research, and portfolio purposes.
