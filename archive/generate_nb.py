import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cell1 = nbf.v4.new_code_cell("""# --- Step 1: Libraries and Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from scipy.sparse import issparse

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
""")
cells.append(cell1)

cell2 = nbf.v4.new_code_cell("""# --- Step 2: Data Loading and Preprocessing ---
data_path = 'data/bug_dataset_50k.csv'
if not os.path.exists(data_path):
    print("Dataset not found. Please ensure it is located at 'data/bug_dataset_50k.csv'")

df = pd.read_csv(data_path)
print(f"Original Dataset Shape: {df.shape}")

# Handle missing values
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
if 'error_code' in df.columns:
    df['error_code'] = df['error_code'].fillna(0)

categorical_cols = ['bug_category', 'tech_stack', 'environment', 'developer_role', 'bug_domain']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str).str.lower().str.strip()

# Merge title and description, and lowercase
df['full_text'] = (df['title'] + " " + df['description']).str.lower()

# Calculate word count for outlier detection
df['desc_word_count'] = df['full_text'].apply(lambda x: len(x.split()))

print("Data preprocessing complete.")
""")
cells.append(cell2)

cell3 = nbf.v4.new_code_cell("""# --- Step 3: Exploratory Data Analysis (EDA) & Outlier Handling ---
print("Class Distribution:")
print(df['severity'].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='severity', order=df['severity'].value_counts().index)
plt.title("Bug Severity Class Distribution")
plt.show()

# Outlier Detection and Handling
plt.figure(figsize=(10, 5))
sns.boxplot(x='severity', y='desc_word_count', data=df)
plt.title("Detecting Outliers in Text Length")
plt.show()

Q1 = df['desc_word_count'].quantile(0.25)
Q3 = df['desc_word_count'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

initial_count = len(df)
df = df[df['desc_word_count'] <= upper_limit].copy()
print(f"Removed {initial_count - len(df)} outlier records based on text length. Remaining records: {len(df)}")
""")
cells.append(cell3)

cell4 = nbf.v4.new_code_cell("""# --- Step 4: Train/Test Split & Feature Engineering ---
# Encode Target
target_le = LabelEncoder()
df['target'] = target_le.fit_transform(df['severity'])

# Separate Features and Target
X = df.drop(columns=['severity', 'target'])
y = df['target']

# Data Leakage Fix: Train/Test Split BEFORE vectorization and encoding
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

# Define ColumnTransformer to handle both categorical and text features safely
transformers = [
    ('cat', OneHotEncoder(handle_unknown='ignore'), [col for col in categorical_cols if col in X.columns]),
    ('text', TfidfVectorizer(max_features=5000, stop_words='english'), 'full_text')
]
if 'error_code' in X.columns:
    transformers.append(('num', 'passthrough', ['error_code']))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

print("Fitting preprocessor and transforming data (producing sparse matrices)...")
X_train_transformed = preprocessor.fit_transform(X_train_raw)
X_test_transformed = preprocessor.transform(X_test_raw)

print(f"Total features after encoding: {X_train_transformed.shape[1]}")

# Feature Selection using Chi-Square
k_features = min(2000, X_train_transformed.shape[1])
print(f"Applying Chi-Square Feature Selection to select top {k_features} features...")

selector = SelectKBest(score_func=chi2, k=k_features)
X_train_selected = selector.fit_transform(X_train_transformed, y_train)
X_test_selected = selector.transform(X_test_transformed)

print(f"Selected feature matrix shape: {X_train_selected.shape}")
""")
cells.append(cell4)

cell5 = nbf.v4.new_code_cell("""# --- Step 5: Model Training and Evaluation ---
models = {
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
}

results = []
best_model = None
best_f1 = 0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\\n--- {name} Performance ---")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Track best model based on F1-Score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results)
print("\\n--- Model Comparison ---")
import IPython.display as display
display.display(results_df)

print(f"\\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
""")
cells.append(cell5)

cell6 = nbf.v4.new_code_cell("""# --- Step 6: Final Output and Prediction Function ---
os.makedirs('models', exist_ok=True)

# Save best model and preprocessors
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(target_le, 'models/target_encoder.pkl')
joblib.dump(selector, 'models/feature_selector.pkl')

print("Assets saved to 'models/' directory.")

def predict_severity(title, description, bug_category='Unknown', tech_stack='Unknown', environment='Unknown', developer_role='Unknown', bug_domain='Unknown', error_code=0):
    # Construct a dataframe with a single row
    input_data = pd.DataFrame([{
        'title': title,
        'description': description,
        'bug_category': bug_category,
        'tech_stack': tech_stack,
        'environment': environment,
        'developer_role': developer_role,
        'bug_domain': bug_domain,
        'error_code': error_code
    }])
    
    # Preprocess exactly as in training
    input_data['full_text'] = (input_data['title'].fillna('') + " " + input_data['description'].fillna('')).str.lower()
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('Unknown').astype(str).str.lower().str.strip()
            
    # Transform
    X_transformed = preprocessor.transform(input_data)
    X_sel = selector.transform(X_transformed)
    
    # Predict
    pred_idx = best_model.predict(X_sel)[0]
    return target_le.inverse_transform([pred_idx])[0]

# Test prediction function
sample_title = "App crashes on login"
sample_desc = "When tapping the login button, the app immediately closes with a fatal exception."
prediction = predict_severity(sample_title, sample_desc)
print(f"\\nSample Prediction for '{sample_title}': {prediction}")
""")
cells.append(cell6)

nb['cells'] = cells
nbf.write(nb, 'bug_severity_pipeline.ipynb')
