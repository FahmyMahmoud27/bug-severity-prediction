# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import joblib
import os
import base64

# ------------------------------------------------------------------
# 1. SETUP & PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Bug Severity Classifier",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 2. ASSET LOADING
# ------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Load the trained machine learning pipeline and target encoder."""
    try:
        # Determine paths relative to this script (app.py is in archive/)
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, '..', 'models')
        
        pipeline = joblib.load(os.path.join(models_dir, 'best_pipeline.pkl'))
        encoder = joblib.load(os.path.join(models_dir, 'target_encoder.pkl'))
        return pipeline, encoder
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.stop()

def load_visual_assets():
    """Load correlation heatmap and chi-square stats."""
    assets = {}
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(base_path, '..')
    
    corr_path = os.path.join(root_dir, 'correlation_matrix.png')
    chi2_path = os.path.join(root_dir, 'chi2_feature_stats.csv')
    
    if os.path.exists(corr_path):
        assets['correlation_matrix'] = corr_path
    
    if os.path.exists(chi2_path):
        assets['chi2_stats'] = pd.read_csv(chi2_path)
        
    return assets

pipeline, target_encoder = load_ml_assets()
visual_assets = load_visual_assets()

# ------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------------
def set_sample(sample_type):
    """Update session state with sample bug data."""
    if sample_type == 'critical':
        st.session_state['title'] = "App crashes on login screen"
        st.session_state['description'] = "When tapping the login button, the application immediately closes with a fatal memory exception. Emergency fix needed."
        st.session_state['bug_category'] = "Crash"
        st.session_state['tech_stack'] = "Mobile"
        st.session_state['environment'] = "Production"
        st.session_state['developer_role'] = "Frontend"
        st.session_state['bug_domain'] = "Auth"
        st.session_state['error_code'] = 500
    elif sample_type == 'low':
        st.session_state['title'] = "Typo in settings menu"
        st.session_state['description'] = "The word 'Preferences' is spelled 'Prefferences' in the user settings dropdown."
        st.session_state['bug_category'] = "UI"
        st.session_state['tech_stack'] = "Frontend"
        st.session_state['environment'] = "Staging"
        st.session_state['developer_role'] = "Designer"
        st.session_state['bug_domain'] = "Settings"
        st.session_state['error_code'] = 0

def predict_severity(input_df):
    """Run data through the pipeline and return prediction with color styling."""
    try:
        pred_idx = pipeline.predict(input_df)[0]
        severity = target_encoder.inverse_transform([pred_idx])[0]
        
        color_map = {
            'Critical': '#FF4B4B', # Red
            'High': '#FFA726',     # Orange
            'Medium': '#FFEB3B',   # Yellow
            'Low': '#4CAF50'       # Green
        }
        color = color_map.get(severity, '#FFFFFF')
        return severity, color
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None, None

# ------------------------------------------------------------------
# 4. UI DESIGN & LAYOUT
# ------------------------------------------------------------------
st.title("🐞 Bug Severity Classification System")
st.markdown("""
Welcome to the automated Bug Triage portal. 
This tool utilizes a highly optimized **Natural Language Processing (NLP)** and **Machine Learning Pipeline** to predict the severity of incoming software bugs.
""")

st.markdown("---")

# Initialize session state for inputs
defaults = {
    'title': '', 'description': '', 'bug_category': 'Unknown', 
    'tech_stack': 'Unknown', 'environment': 'Unknown', 
    'developer_role': 'Unknown', 'bug_domain': 'Unknown', 'error_code': 0
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Quick Test Buttons
st.markdown("### Quick Tests")
col_s1, col_s2, _ = st.columns([1, 1, 4])
with col_s1:
    if st.button("🚨 Load Critical Bug Sample"):
        set_sample('critical')
with col_s2:
    if st.button("✅ Load Low Bug Sample"):
        set_sample('low')

st.markdown("---")

# Main Input Form
st.markdown("### Bug Details")
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Bug Title *", key='title')
    description = st.text_area("Bug Description *", key='description', height=180)
    
with col2:
    bug_category = st.selectbox("Bug Category", ["Unknown", "Crash", "UI", "Performance", "Security", "Network"], key='bug_category')
    tech_stack = st.selectbox("Tech Stack", ["Unknown", "Mobile", "Frontend", "Backend", "Database", "DevOps"], key='tech_stack')
    environment = st.selectbox("Environment", ["Unknown", "Production", "Staging", "Development", "Local"], key='environment')
    developer_role = st.selectbox("Assignee Role", ["Unknown", "Frontend", "Backend", "Designer", "QA", "Security"], key='developer_role')
    bug_domain = st.selectbox("Bug Domain", ["Unknown", "Auth", "Settings", "Payments", "Data", "Core"], key='bug_domain')
    error_code = st.number_input("Error Code (0 if none)", value=0, step=1, key='error_code')

st.markdown("---")

# ------------------------------------------------------------------
# 5. PREDICTION & INSIGHTS
# ------------------------------------------------------------------
if st.button("🔮 Predict Severity", type="primary", use_container_width=True):
    if not title.strip() or not description.strip():
        st.warning("⚠️ Please provide both a Title and a Description.")
    else:
        with st.spinner("Analyzing..."):
            # Prepare Input
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
            
            # Feature Engineering for Prediction
            input_data['full_text'] = (input_data['title'].fillna('') + " " + input_data['description'].fillna('')).str.lower()
            input_data['desc_word_count'] = input_data['description'].apply(lambda x: len(str(x).split()))
            
            # Predict
            severity, color = predict_severity(input_data)
            
            if severity:
                st.markdown(f'''
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h2 style="color: #1E1E1E; margin:0;">Predicted Severity: {severity.upper()}</h2>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.info(f"**Methodology Insight:** The model identified key patterns in the textual content and metadata. Bugs mentioning keywords like 'crash' or appearing in 'Production' are prioritized.")

                # Display Insights in Tabs
                tab1, tab2, tab3 = st.tabs(["Feature Importance", "Correlation Analysis", "Statistical Significance"])
                
                with tab1:
                    st.subheader("Top 10 Influential Features")
                    clf = pipeline.named_steps['clf']
                    if hasattr(clf, 'feature_importances_'):
                        try:
                            # Retrieve feature names
                            preprocessor = pipeline.named_steps['preprocessor']
                            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                            cat_features = ohe.get_feature_names_out()
                            tfidf_feat = preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
                            num_features = preprocessor.transformers_[0][2]
                            all_features = list(num_features) + list(cat_features) + list(tfidf_feat)
                            
                            selected_idx = pipeline.named_steps['chi2'].get_support(indices=True)
                            selected_features = [all_features[i] for i in selected_idx]
                            
                            importances = clf.feature_importances_
                            imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
                            top_imp = imp_df.sort_values('Importance', ascending=False).head(10)
                            
                            st.bar_chart(top_imp.set_index('Feature'))
                        except Exception as e:
                            st.write("Could not render feature importance chart.")
                    else:
                        st.write("Feature importance not available for this model type (e.g., Naive Bayes/Logistic Regression).")

                with tab2:
                    st.subheader("Numeric Feature Correlation")
                    if 'correlation_matrix' in visual_assets:
                        st.image(visual_assets['correlation_matrix'], use_column_width=True)
                    else:
                        st.write("Correlation matrix image not found. Run the notebook script first.")

                with tab3:
                    st.subheader("Chi-Square Feature Significance (Categorical)")
                    if 'chi2_stats' in visual_assets:
                        st.dataframe(visual_assets['chi2_stats'].sort_values('chi2_score', ascending=False).head(20), use_container_width=True)
                    else:
                        st.write("Chi-square statistics not found. Run the notebook script first.")
