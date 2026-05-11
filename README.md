# 🐞 Bug Severity Prediction System

An end-to-end Machine Learning project that predicts the severity of software bugs based on bug reports, metadata, and textual descriptions.  
The project is built with a production-ready pipeline and includes a full Data Science workflow from preprocessing to deployment using Streamlit.

---

## 🚀 Project Overview

This system analyzes bug reports and automatically classifies their severity level (e.g., Low, Medium, High, Critical) using Machine Learning models.  
It combines NLP techniques and structured data analysis to build an intelligent classification system.

---

## 📊 Dataset

- Source: `bug_dataset_50k.csv`
- Contains:
  - Bug title & description
  - Metadata (category, error codes, etc.)
  - Severity labels

---

## 🧠 Key Features

### 🔹 Data Preprocessing
- Missing value handling
- Categorical encoding
- Feature scaling
- Text cleaning & normalization

### 🔹 Exploratory Data Analysis (EDA)
- Data distribution analysis
- Class imbalance visualization
- Correlation heatmap
- Feature insights

### 🔹 Feature Engineering
- TF-IDF for text features
- Word count & metadata features
- Feature selection techniques

### 🔹 Statistical Analysis
- Chi-Square test for categorical features
- Correlation analysis for numerical features
- Outlier detection using IQR method

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression (Baseline)
- Naive Bayes (MultinomialNB)
- Random Forest Classifier
- XGBoost Classifier

---

## 📈 Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score (Macro)
- Confusion Matrix
- Cross-validation

📌 Final model selected based on highest Macro-F1 score.

---

## 🏆 Best Model

The best-performing model is saved as:


It is used for real-time predictions in the Streamlit app.

---

## 🌐 Streamlit Web App

A fully interactive UI built using Streamlit allows users to:

- Input bug report details
- Get real-time severity prediction
- View prediction probabilities
- Explore feature importance
- Visualize model insights

### ▶️ Run the app:

```bash
streamlit run app.py

📁 Project Structure
DATAproject/
│
├── data/
├── models/
├── archive/
├── bug_severity_pipeline.ipynb
├── end_to_end_bug_severity_project.ipynb
├── generate_end_to_end_nb.py
├── requirements.txt
├── run_project.bat
└── app.py (Streamlit UI)

⚙️ Installation
git clone https://github.com/USERNAME/bug-severity-prediction.git
cd bug-severity-prediction

pip install -r requirements.txt

📌 Tech Stack
Python 🐍
Pandas / NumPy
Scikit-learn
XGBoost
SciPy
Matplotlib / Seaborn
Streamlit
💡 Business Impact
Reduces time needed to classify bug severity
Helps developers prioritize critical issues
Improves software maintenance efficiency
Automates manual bug triaging process
📌 Future Improvements
Deep Learning-based text classification (LSTM / Transformers)
Deployment on cloud (AWS / Azure)
CI/CD pipeline integration
Real-time bug tracking integration
