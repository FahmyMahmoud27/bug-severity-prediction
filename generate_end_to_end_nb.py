import nbformat as nbf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Scikit‑learn imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.feature_selection import chi2, SelectKBest

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # ---------- Introduction ----------
    cells.append(nbf.v4.new_markdown_cell("""
# End‑to‑End Bug Severity Prediction Project

## 1. Project Overview
This notebook implements a complete, reproducible pipeline that predicts bug severity from textual and categorical metadata. It covers:
1. Data Loading & EDA
2. Outlier Detection (IQR)
3. Correlation Analysis
4. Feature Engineering
5. Chi‑Square feature selection
6. Model training (Logistic Regression, Random Forest, XGBoost, Naive Bayes)
7. Hyper‑parameter tuning for XGBoost
8. Model selection based on macro‑F1
9. Final pipeline serialization
10. Feature importance visualization
"""))

    # ---------- Data Loading ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 2. Data Loading & Initial Exploration
"""))
    cells.append(nbf.v4.new_code_cell("""
data_path = 'Data/bug_dataset_50k.csv'
df = pd.read_csv(data_path)
print('Dataset shape:', df.shape)
df.head()
"""))

    # ---------- Deterministic Severity Logic ----------
    cells.append(nbf.v4.new_markdown_cell("""
### 2.1 Re‑inject deterministic severity logic (if needed)
"""))
    cells.append(nbf.v4.new_code_cell("""
def assign_logic(row):
    text = f"{row.get('title','')} {row.get('description','')}".lower()
    if any(w in text for w in ['crash', 'security', 'leak', 'fatal', 'vulnerability']):
        return 'Critical'
    if 'production' in str(row.get('environment','')).lower() or any(w in text for w in ['api', 'database', 'timeout']):
        return 'High'
    if 'staging' in str(row.get('environment','')).lower() or 'backend' in str(row.get('developer_role','')).lower():
        return 'Medium'
    return 'Low'

df['severity'] = df.apply(assign_logic, axis=1)
"""))

    # ---------- Outlier Detection (IQR) ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 3. Outlier Detection (IQR) on word count
"""))
    cells.append(nbf.v4.new_code_cell("""
# Create combined text and word count
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
df['full_text'] = (df['title'] + ' ' + df['description']).str.lower()
df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))

Q1 = df['desc_word_count'].quantile(0.25)
Q3 = df['desc_word_count'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
orig_len = len(df)
df = df[(df['desc_word_count'] >= lower) & (df['desc_word_count'] <= upper)]
print(f'Removed {orig_len - len(df)} outlier rows based on word count')
"""))

    # ---------- Correlation Analysis ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 4. Correlation Analysis (numeric features)
"""))
    cells.append(nbf.v4.new_code_cell("""
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()
"""))

    # ---------- Feature Engineering ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 5. Feature Engineering
"""))
    cells.append(nbf.v4.new_code_cell("""
# Drop identifiers & unused columns
drop_cols = ['bug_id', 'created_at', 'title', 'description', 'suggested_fix', 'explanation', 'root_cause']
df_ml = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target encoding
le_target = LabelEncoder()
y = le_target.fit_transform(df_ml['severity'])
X = df_ml.drop(columns=['severity'])
print('Class mapping:', dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))
"""))

    # ---------- Preprocessing & Chi‑Square ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 6. Preprocessing (ColumnTransformer) + Chi‑Square feature selection
"""))
    cells.append(nbf.v4.new_code_cell("""
text_feat = 'full_text'
numeric_feats = ['error_code', 'desc_word_count']
categorical_feats = ['bug_category', 'bug_domain', 'tech_stack', 'environment', 'developer_role']

numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', MinMaxScaler())])
categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                             ('onehot', OneHotEncoder(handle_unknown='ignore'))])
text_pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=500, stop_words='english'))])

preprocessor = ColumnTransformer([('num', numeric_pipe, numeric_feats),
                               ('cat', categorical_pipe, categorical_feats),
                               ('text', text_pipe, text_feat)])

chi_selector = SelectKBest(score_func=chi2, k=2000)
"""))

    # ---------- Train/Test Split ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 7. Train/Test Split
"""))
    cells.append(nbf.v4.new_code_cell("""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
"""))

    # ---------- Model Building ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 8. Model Building & Baseline Comparison
"""))
    cells.append(nbf.v4.new_code_cell("""
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Naive Bayes': MultinomialNB()
}

baseline_results = {}
for name, model in models.items():
    print(f'Training {name}...')
    pipe = Pipeline([('preprocessor', preprocessor), ('chi2', chi_selector), ('clf', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    baseline_results[name] = {
        'pipeline': pipe, 
        'Accuracy': accuracy_score(y_test, preds), 
        'Macro_F1': f1_score(y_test, preds, average='macro'), 
        'Preds': preds
    }
"""))

    # ---------- Hyperparameter Tuning ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 9. Hyperparameter Tuning (XGBoost)
"""))
    cells.append(nbf.v4.new_code_cell("""
xgb_pipe = Pipeline([('preprocessor', preprocessor), ('chi2', chi_selector), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))])
param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5], 'clf__learning_rate': [0.1, 0.2]}
search = RandomizedSearchCV(xgb_pipe, param_grid, n_iter=3, scoring='f1_macro', cv=2, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
"""))

    # ---------- Selection & Saving ----------
    cells.append(nbf.v4.new_markdown_cell("""
## 10. Model Selection & Artifact Generation
"""))
    cells.append(nbf.v4.new_code_cell("""
best_xgb = search.best_estimator_
preds_xgb = best_xgb.predict(X_test)
baseline_results['XGBoost (Tuned)'] = {
    'pipeline': best_xgb, 
    'Accuracy': accuracy_score(y_test, preds_xgb),
    'Macro_F1': f1_score(y_test, preds_xgb, average='macro'), 
    'Preds': preds_xgb
}

best_name = max(baseline_results, key=lambda k: baseline_results[k]['Macro_F1'])
best_model = baseline_results[best_name]['pipeline']

# Chi-Square Stats
X_train_pre = preprocessor.fit_transform(X_train, y_train)
chi_scores, chi_pvals = chi2(X_train_pre, y_train)
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
tfidf_feat = preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
all_features = list(numeric_feats) + list(ohe.get_feature_names_out()) + list(tfidf_feat)
chi_df = pd.DataFrame({'feature': all_features, 'chi2_score': chi_scores, 'p_value': chi_pvals})
chi_df.to_csv('chi2_feature_stats.csv', index=False)

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_pipeline.pkl')
joblib.dump(le_target, 'models/target_encoder.pkl')
"""))

    nb['cells'] = cells
    with open('end_to_end_bug_severity_project.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('Notebook generated: end_to_end_bug_severity_project.ipynb')

def run_full_pipeline():
    print("Executing full ML pipeline to generate artifacts...")
    data_path = 'Data/bug_dataset_50k.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # 1. Logic & Cleaning
    def assign_logic(row):
        text = f"{row.get('title','')} {row.get('description','')}".lower()
        if any(w in text for w in ['crash', 'security', 'leak', 'fatal', 'vulnerability']): return 'Critical'
        if 'production' in str(row.get('environment','')).lower() or any(w in text for w in ['api', 'database', 'timeout']): return 'High'
        if 'staging' in str(row.get('environment','')).lower() or 'backend' in str(row.get('developer_role','')).lower(): return 'Medium'
        return 'Low'
    df['severity'] = df.apply(assign_logic, axis=1)
    
    df['description'] = df['description'].fillna('')
    df['full_text'] = (df['title'].fillna('') + ' ' + df['description']).str.lower()
    df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    
    # 2. IQR
    Q1, Q3 = df['desc_word_count'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df['desc_word_count'] >= Q1 - 1.5*IQR) & (df['desc_word_count'] <= Q3 + 1.5*IQR)]
    
    # 3. Correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # 4. ML Pipeline
    le = LabelEncoder()
    y = le.fit_transform(df['severity'])
    X = df.drop(columns=['severity'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_feats = ['error_code', 'desc_word_count']
    cat_feats = ['bug_category', 'bug_domain', 'tech_stack', 'environment', 'developer_role']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]), num_feats),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_feats),
        ('text', Pipeline([('tfidf', TfidfVectorizer(max_features=500, stop_words='english'))]), 'full_text')
    ])
    
    chi_selector = SelectKBest(chi2, k=2000)
    
    # Use XGBoost as the "best" for artifact generation speed in script
    final_pipe = Pipeline([('preprocessor', preprocessor), ('chi2', chi_selector), ('clf', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss'))])
    final_pipe.fit(X_train, y_train)
    
    # Chi-stats
    X_train_pre = preprocessor.transform(X_train)
    scores, pvals = chi2(X_train_pre, y_train)
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    tfidf = preprocessor.named_transformers_['text'].named_steps['tfidf']
    all_cols = num_feats + list(ohe.get_feature_names_out()) + list(tfidf.get_feature_names_out())
    pd.DataFrame({'feature': all_cols, 'chi2_score': scores, 'p_value': pvals}).to_csv('chi2_feature_stats.csv', index=False)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_pipe, 'models/best_pipeline.pkl')
    joblib.dump(le, 'models/target_encoder.pkl')
    print("Pipeline execution complete. Artifacts generated.")

if __name__ == "__main__":
    create_notebook()
    run_full_pipeline()
