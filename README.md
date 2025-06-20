# Predicting-Diabetes-Risk-Using-Machine-Learning-A-Comparative-Analysis
🔍 Diabetes Prediction Using Machine Learning This project uses the Pima Indians Diabetes dataset to build and compare classification models—Logistic Regression, Random Forest, and XGBoost—for predicting diabetes risk. It includes data cleaning, model evaluation, feature importance analysis, and insights to support early medical intervention.
Here’s a professional and detailed **GitHub repository description (README.md content)** for your diabetes classification project:

---

# 🩺 Diabetes Prediction Using Machine Learning

This project focuses on predicting the likelihood of diabetes in individuals using various classification algorithms. Using the Pima Indians Diabetes dataset, we explore, model, and interpret medical data to assist healthcare professionals in identifying high-risk patients for early intervention.

---

## 🚀 Project Objective

To build and compare machine learning classification models that can accurately predict diabetes based on patient medical attributes. The aim is both **predictive accuracy** and **model interpretability** to inform real-world clinical decision-making.

---

## 📊 Dataset

**Source:** Pima Indians Diabetes Dataset from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

**Target:**

* `Outcome` — 1: Diabetic, 0: Non-Diabetic

---

## 🛠️ Workflow

### 1. **Data Preprocessing**

* Replaced invalid 0s with NaNs in medical columns.
* Imputed missing values using **median**.
* Scaled features using `StandardScaler`.

### 2. **Exploratory Data Analysis**

* Correlation matrix
* Feature distribution plots
* Class distribution analysis

### 3. **Model Building**

Trained the following classifiers:

* **Logistic Regression** — interpretable baseline
* **Random Forest Classifier** — robust ensemble model
* **XGBoost Classifier** — high-performance gradient boosting

### 4. **Model Evaluation**

* Accuracy, Precision, Recall, F1 Score
* ROC AUC and Precision-Recall curves
* Feature importance analysis
* SHAP values for model explainability

---

## 🧠 Results & Findings

* **XGBoost** achieved the highest ROC AUC, indicating strong predictive performance.
* **Top Features:** Glucose, BMI, Age, and Insulin.
* **Trade-offs:** Logistic Regression offered better interpretability, while XGBoost provided better accuracy and recall.

---

## 📉 Limitations

* Dataset lacks lifestyle and hereditary attributes.
* Moderate class imbalance not fully addressed.
* Only basic imputation and scaling used — further feature engineering could help.

---

## 🔮 Future Work

* Handle class imbalance with **SMOTE/ADASYN**.
* Collect additional features (e.g., diet, activity level).
* Explore deep learning and ensemble stacking.
* Deploy as a web app using Streamlit or Flask.

---

## 📂 Repository Structure

```
📁 Diabetes-Prediction-ML
├── Diabetes.ipynb             # Main notebook
│── diabetes.csv           # Dataset
├── README.md                  # Project overview

```

---

## ✅ Requirements

* Python 3.8+
* pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, shap

Install using:

```bash
pip install -r requirements.txt
```

---

## 📬 Contact

For questions or collaborations:

**\[Your Name]**
Email: [your.email@example.com](mailto:irakamsivabhanuprakash@gmail.com)
LinkedIn: \[Your LinkedIn Profile](https://www.linkedin.com/in/siva-venkata-bhanu-prakash/)
GitHub: \[Your GitHub Handle](https://github.com/hynko431/)

---

Would you like me to generate a `README.md` file and upload it here for direct use in your repo?
