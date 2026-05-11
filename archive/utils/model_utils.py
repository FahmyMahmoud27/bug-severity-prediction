import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline and target encoder."""
    try:
        # Get absolute path relative to the app execution directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        pipeline_path = os.path.join(base_dir, 'models', 'best_pipeline.pkl')
        encoder_path = os.path.join(base_dir, 'models', 'target_encoder.pkl')
        
        pipeline = joblib.load(pipeline_path)
        encoder = joblib.load(encoder_path)
        return pipeline, encoder
    except Exception as e:
        st.error(f"Error loading model assets. Please check if files exist in 'models/': {e}")
        return None, None

def predict_severity(input_df, pipeline, target_encoder):
    """Run data through the pipeline and return prediction with confidence and styling."""
    try:
        # Preprocessing inputs consistently with training
        input_data = input_df.copy()
        input_data['full_text'] = (input_data['title'].fillna('') + " " + input_data['description'].fillna('')).str.lower()
        
        categorical_cols = ['bug_category', 'tech_stack', 'environment', 'developer_role', 'bug_domain']
        for col in categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].fillna('Unknown').astype(str).str.lower().str.strip()

        # Predict class index
        pred_idx = pipeline.predict(input_data)[0]
        # Decode index to string
        severity = target_encoder.inverse_transform([pred_idx])[0]
        
        # Get confidence score if possible
        confidence = None
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_data)[0]
            confidence = np.max(probs) * 100
        
        # Color mapping for UI
        color_map = {
            'Critical': '#FF4B4B', # Red
            'High': '#FFA726',     # Orange
            'Medium': '#FFEB3B',   # Yellow
            'Low': '#4CAF50'       # Green
        }
        color = color_map.get(severity, '#FFFFFF')
        
        # Generate simple explanation based on keywords
        explanation = generate_explanation(input_data.iloc[0], severity)
        
        return {
            "severity": severity,
            "confidence": confidence,
            "color": color,
            "explanation": explanation
        }
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

def generate_explanation(row, severity):
    """Generate a simple explanation based on found keywords and metadata."""
    text = row.get('full_text', '')
    env = row.get('environment', '')
    
    keywords_found = []
    critical_keywords = ['crash', 'fatal', 'emergency', 'memory', 'exception', 'vulnerability']
    
    for kw in critical_keywords:
        if kw in text:
            keywords_found.append(kw)
            
    if severity in ['Critical', 'High']:
        reason = f"The model flagged this as **{severity}** "
        if keywords_found:
            reason += f"due to the presence of critical keywords like: `{', '.join(keywords_found)}`. "
        if env == 'production':
            reason += "Bugs reported in the **production** environment are often elevated in severity."
    else:
        reason = f"The model flagged this as **{severity}** because it lacks aggressive keywords like 'crash' or 'fatal', and seems to describe a non-blocking issue."
        
    return reason
