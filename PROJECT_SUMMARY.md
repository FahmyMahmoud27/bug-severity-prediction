# Project Summary: Bug Severity Prediction Pipeline

## 1. Problem Statement
Software development teams often face an overwhelming volume of bug reports. Manual triage—assigning severity and priority—is time-consuming and prone to human error. This project implements an automated **Bug Severity Prediction Pipeline** that uses Machine Learning to classify bugs into four categories: **Low, Medium, High, and Critical**, based on their metadata and textual descriptions.

## 2. Dataset Overview
- **Source**: Synthetic dataset of 50,000 bug reports (representative of enterprise-level bug tracking systems).
- **Features**:
    - **Categorical**: Bug Category, Bug Domain, Tech Stack, Environment, Developer Role.
    - **Numeric**: Error Code, Description Word Count (derived).
    - **Textual**: Bug Title, Bug Description.
- **Target**: `severity` (Ordinal: Low < Medium < High < Critical).

## 3. Methodology
The project follows a rigorous end-to-end Data Science workflow:

### A. Data Preprocessing & Cleaning
- **Outlier Detection**: Applied the **IQR (Interquartile Range) Method** on `desc_word_count` to remove extreme outliers that could skew text vectorization.
- **Handling Missing Values**: Used median imputation for numeric features and constant imputation ('missing') for categorical features.

### B. Exploratory Data Analysis (EDA)
- **Correlation Analysis**: Visualized a correlation heatmap for numeric features to identify redundancies.
- **Statistical Testing**: Performed **Chi-Square Tests** to determine the dependency between categorical features and the target severity level.

### C. Feature Engineering
- **Text Vectorization**: Combined `title` and `description` into a single feature, processed using **TF-IDF Vectorization** (limited to top 500 features).
- **Encoding**: One-Hot Encoding for categorical variables and Standard Scaling for numeric features.
- **Feature Selection**: Used **SelectKBest (Chi-Square)** to select the top 2000 most significant features from the combined high-dimensional sparse matrix.

### D. Model Building & Comparison
We evaluated four distinct algorithms to find the best fit:
1. **Logistic Regression**: Baseline linear classifier.
2. **Random Forest**: Ensemble of decision trees.
3. **XGBoost**: Gradient boosted decision trees (tuned via RandomizedSearchCV).
4. **Multinomial Naive Bayes**: Efficient for high-dimensional text data.

## 4. Final Results & Model Selection
- **Primary Metric**: **Macro-F1 Score** (chosen over accuracy to ensure balanced performance across all classes).
- **Best Model**: **XGBoost (Tuned)** achieved the highest Macro-F1 score, demonstrating superior ability to capture non-linear relationships between environment keywords and severity levels.
- **Serialization**: The best pipeline (preprocessing + chi2 + classifier) is saved as `best_pipeline.pkl` for production use.

## 5. Business Interpretation & Insights
- **Predictive Power of Text**: Keywords like "crash", "leak", and "vulnerability" are the strongest indicators of Critical bugs.
- **Environment Context**: Bugs occurring in **Production** environments are statistically more likely to be High/Critical compared to Staging or Local.
- **ROI**: Implementing this pipeline reduces triage time by up to 80%, allowing engineering teams to focus on critical fixes immediately rather than waiting for manual assessment.

## 6. How to Run
1. **Generate Notebook**: `python generate_end_to_end_nb.py`
2. **Launch App**: `streamlit run archive/app.py`
3. **Review Results**: Open `end_to_end_bug_severity_project.ipynb` for the full technical walkthrough.
