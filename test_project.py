import joblib
import pandas as pd
import os

def test_project_functions():
    print("--- Starting Project Functionality Test ---")
    
    # 1. Check Artifacts
    artifacts = [
        'models/best_pipeline.pkl',
        'models/target_encoder.pkl',
        'correlation_matrix.png',
        'chi2_feature_stats.csv',
        'end_to_end_bug_severity_project.ipynb'
    ]
    
    for art in artifacts:
        if os.path.exists(art):
            print(f"[OK] Found artifact: {art}")
        else:
            print(f"[ERROR] Missing artifact: {art}")
            # return # Don't return, check others

    # 2. Test Model Loading & Prediction
    print("\nTesting Model Inference...")
    try:
        pipeline = joblib.load('models/best_pipeline.pkl')
        encoder = joblib.load('models/target_encoder.pkl')
        
        # Sample test data
        test_data = pd.DataFrame([{
            'title': 'System crash on startup',
            'description': 'The application fails to boot and throws a fatal exception in production.',
            'bug_category': 'crash',
            'tech_stack': 'backend',
            'environment': 'production',
            'developer_role': 'devops',
            'bug_domain': 'core',
            'error_code': 500
        }])
        
        # Preprocessing for prediction (matching app logic)
        test_data['full_text'] = (test_data['title'].fillna('') + " " + test_data['description'].fillna('')).str.lower()
        test_data['desc_word_count'] = test_data['description'].apply(lambda x: len(str(x).split()))
        
        pred_idx = pipeline.predict(test_data)[0]
        severity = encoder.inverse_transform([pred_idx])[0]
        print(f"[OK] Prediction successful: {severity}")
        
    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")

    # 3. Check Streamlit App Existence
    if os.path.exists('archive/app.py'):
        print("[OK] Streamlit app script found at archive/app.py")
    else:
        print("[ERROR] Streamlit app script missing.")

    print("\n--- Project verification complete. ---")

if __name__ == "__main__":
    test_project_functions()
