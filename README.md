# CodSoft Projects

## 📌 Overview
This repository contains multiple machine learning projects completed under the CodSoft initiative.
The projects demonstrate practical applications of data preprocessing, model building, and evaluation across different domains:

1. Churn Rate Prediction – Predicts whether a customer will leave a service.
2. Movie Genre Classification – Classifies movies into genres based on metadata or text descriptions.
3. SMS Spam Prediction – Detects whether an SMS message is spam or ham.

All projects are designed to run locally and are implemented in Python using popular libraries like scikit-learn, pandas, NumPy, and TensorFlow/Keras where applicable.

---

## 🎯 Key Features
- Practical, end-to-end machine learning pipelines
- Data preprocessing, feature engineering, and encoding
- Multiple model experimentation
- Evaluation using standard metrics (accuracy, precision, recall, F1-score)
- Local execution with clear instructions
- Well-documented and modular code

---

## 🧠 Projects

### 1. Churn Rate Prediction
Objective: Predict whether a customer is likely to churn.
Tech Stack & Models:
- Preprocessing: Missing value imputation, One-Hot Encoding, Standard Scaling
- Models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Confusion matrix, ROC-AUC, Accuracy
Dataset: Telecom / simulated customer data

---

### 2. Movie Genre Classification
Objective: Classify movies into genres based on metadata or textual description.
Tech Stack & Models:
- Preprocessing: Text cleaning, TF-IDF vectorization, Label Encoding
- Models: Multinomial Naive Bayes, Random Forest, Neural Networks
- Evaluation: Accuracy, Precision, Recall, F1-score
Dataset: MovieLens / IMDB dataset

---

### 3. SMS Spam Prediction
Objective: Detect whether an SMS message is spam or ham.
Tech Stack & Models:
- Preprocessing: Text cleaning, Tokenization, TF-IDF vectorization
- Models: Logistic Regression, Random Forest, Naive Bayes
- Evaluation: Confusion matrix, Accuracy, Precision, Recall, F1-score
Dataset: UCI SMS Spam Collection

---


## 🚀 How to Run Locally
1. Clone the Repository
git clone https://github.com/your-username/codsoft-projects.git
cd codsoft-projects

2. Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r <project-folder>/requirements.txt

4. Run the Jupyter Notebook
jupyter notebook
- Open the notebook for the desired project (Churn / Movie Genre / SMS Spam)
- Execute cells step by step

---

## 🧪 Workflow in All Projects
1. Load and explore dataset
2. Handle missing data and clean features
3. Encode categorical features and scale numerical features
4. Split data into train and test sets
5. Train machine learning models
6. Evaluate performance using metrics
7. Optionally save trained models for deployment

---

## 📊 Output
- Churn Rate Prediction → Churn probability for each customer
- Movie Genre Classification → Predicted genres per movie
- SMS Spam Prediction → Spam or Ham prediction for input messages

