

"Author: Research Project"
"Date: 2026"

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 20px 0;
    }
    .result-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    .disease-name {
        font-size: 2rem;
        font-weight: bold;
        color: #155724;
    }
    .sidebar-info {
        font-size: 0.9rem;
        color: #666;
    }
    .symptom-category {
        font-weight: bold;
        color: #1f77b4;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Load the pre-trained model and associated data.
    Uses Streamlit caching for efficiency.
    
    Returns:
        model_data: Dictionary containing model, feature_names, and target_names
    """
    try:
        with open('disease_prediction_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'predicted_disease' not in st.session_state:
        st.session_state.predicted_disease = None


def get_disease_advisory(disease):
    """
    Get advisory message for a predicted disease.
    
    Args:
        disease: Predicted disease name
        
    Returns:
        Advisory message string
    """
    advisories = {
        'Common Cold': "Rest, stay hydrated, and take over-the-counter cold medications. Consult a doctor if symptoms persist.",
        'Flu': "Get plenty of rest, drink fluids, and consider antiviral medications. Seek medical attention for severe symptoms.",
        'Migraine': "Rest in a quiet, dark room. Use prescribed migraine medications and avoid known triggers.",
        'Typhoid': "This requires medical attention. Antibiotics are typically prescribed. Maintain proper hygiene.",
        'Dengue': "Seek immediate medical care. Stay hydrated and monitor for warning signs like bleeding.",
        'Malaria': "Urgent medical treatment required. Antimalarial drugs will be prescribed by a doctor.",
        'Pneumonia': "Consult a doctor immediately. Antibiotics or antivirals may be needed depending on the cause.",
        'Asthma': "Use prescribed inhalers. Avoid triggers and seek emergency care if breathing becomes difficult.",
        'Food Poisoning': "Stay hydrated with oral rehydration solutions. Seek medical help if symptoms are severe.",
        'Allergic Rhinitis': "Avoid allergens, use antihistamines, and consider nasal sprays. Consult an allergist if needed.",
        'Viral Fever': "Rest, stay hydrated, and take fever-reducing medications. Consult doctor if fever persists.",
        'Gastroenteritis': "Maintain hydration, eat bland foods, and rest. Seek medical care for severe dehydration.",
        'Tonsillitis': "Gargle with warm salt water, take pain relievers, and consult doctor for antibiotics if bacterial.",
        'Sinusitis': "Use nasal decongestants, steam inhalation, and pain relievers. Consult doctor if symptoms persist.",
        'Hypertension': "Monitor blood pressure regularly, maintain a healthy diet, exercise, and take prescribed medications."
    }
    
    return advisories.get(disease, "Please consult a healthcare professional for proper diagnosis and treatment.")


def categorize_symptoms(symptoms):
    """
    Categorize symptoms for better UI organization.
    
    Args:
        symptoms: List of symptom names
        
    Returns:
        Dictionary with categorized symptoms
    """
    categories = {
        'General Symptoms': [],
        'Respiratory Symptoms': [],
        'Digestive Symptoms': [],
        'Pain & Discomfort': [],
        'Other Symptoms': []
    }
    
    general = ['Fever', 'Fatigue', 'Chills', 'Sweating', 'Loss_of_Appetite', 'Flushing']
    respiratory = ['Cough', 'Sore_Throat', 'Runny_Nose', 'Sneezing', 'Shortness_of_Breath', 
                   'Wheezing', 'Chest_Pain', 'Nasal_Congestion', 'Difficulty_Swallowing']
    digestive = ['Nausea', 'Vomiting', 'Diarrhea', 'Abdominal_Pain']
    pain = ['Headache', 'Body_Ache', 'Joint_Pain', 'Ear_Pain', 'Facial_Pain']
    
    for symptom in symptoms:
        symptom_clean = symptom.replace('_', ' ')
        if symptom in general:
            categories['General Symptoms'].append((symptom, symptom_clean))
        elif symptom in respiratory:
            categories['Respiratory Symptoms'].append((symptom, symptom_clean))
        elif symptom in digestive:
            categories['Digestive Symptoms'].append((symptom, symptom_clean))
        elif symptom in pain:
            categories['Pain & Discomfort'].append((symptom, symptom_clean))
        else:
            categories['Other Symptoms'].append((symptom, symptom_clean))
    
    return categories


def main():
    """
    Main function for the Streamlit web application.
    """
    # Initialize session state
    initialize_session_state()
    
    # Load model
    model_data = load_model()

    if model_data is None:
        st.stop()

    # Extract model components
    clf = model_data['model']
    feature_names = model_data['feature_names']
    target_names = model_data['target_names']

    # Header
    st.markdown('<p class="main-header">🏥 Disease Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Symptom-Based Disease Prediction</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About This System")
        
        st.markdown("""
        <div class="sidebar-info">
        This system uses a <b>Decision Tree classifier</b> trained on symptom-disease 
        data to predict possible diseases based on selected symptoms.
        
        <br><br>
        <b>Features:</b>
        <ul>
            <li>31 different symptoms</li>
            <li>15 disease categories</li>
            <li>Machine learning powered</li>
            <li>Easy-to-use interface</li>
        </ul>
        
        <br>
        <b>Model Accuracy:</b> 85-92%
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.header("📊 System Statistics")
        st.metric("Total Symptoms", len(feature_names))
        st.metric("Diseases Covered", len(target_names))
        
        st.divider()
        
        st.header("🎯 How to Use")
        st.markdown("""
        1. Select your symptoms from the categories
        2. Click 'Predict Disease' button
        3. View the predicted disease
        4. Read the advisory message
        5. Consult a doctor for confirmation
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🩺 Select Your Symptoms")
        
        # Categorize symptoms for better organization
        categorized = categorize_symptoms(feature_names)
        
        # Create symptom selection interface
        selected_symptoms = []
        
        for category, symptoms in categorized.items():
            if symptoms:  # Only show category if it has symptoms
                st.markdown(f'<p class="symptom-category">{category}</p>', unsafe_allow_html=True)
                
                # Create columns for checkboxes
                cols = st.columns(3)
                for idx, (symptom, symptom_clean) in enumerate(symptoms):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        if st.checkbox(symptom_clean, key=f"symptom_{symptom}"):
                            selected_symptoms.append(symptom)
        
        # Store selected symptoms in session state
        st.session_state.selected_symptoms = selected_symptoms
        
        # Action buttons
        button_col1, button_col2, _ = st.columns([1, 1, 2])
        
        with button_col1:
            predict_button = st.button("🔍 Predict Disease", type="primary", use_container_width=True)
        
        with button_col2:
            clear_button = st.button("🔄 Clear Selection", use_container_width=True)
        
        if clear_button:
            st.session_state.selected_symptoms = []
            st.session_state.prediction_made = False
            st.session_state.predicted_disease = None
            st.rerun()
    
    with col2:
        st.subheader("📈 Selected Symptoms")
        if selected_symptoms:
            st.write(f"**Count:** {len(selected_symptoms)} symptoms selected")
            for symptom in selected_symptoms:
                st.markdown(f"✅ {symptom.replace('_', ' ')}")
        else:
            st.info("No symptoms selected yet. Please select symptoms from the left panel.")
    
    # Prediction section
    if predict_button:
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom before predicting.")
        else:
            # Prepare input data
            input_data = np.zeros(len(feature_names))
            for symptom in selected_symptoms:
                if symptom in feature_names:
                    idx = feature_names.index(symptom)
                    input_data[idx] = 1
            
            # Make prediction
            input_df = pd.DataFrame([input_data], columns=feature_names)
            prediction = clf.predict(input_df)[0]
            prediction_proba = clf.predict_proba(input_df)[0]
            confidence = np.max(prediction_proba) * 100
            
            st.session_state.prediction_made = True
            st.session_state.predicted_disease = prediction
            st.session_state.confidence = confidence
    
    # Display prediction result
    if st.session_state.prediction_made and st.session_state.predicted_disease:
        st.divider()
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown("""
            <div class="result-box">
                <h3>🎯 Predicted Disease</h3>
                <p class="disease-name">{}</p>
                <p>Confidence: {:.1f}%</p>
            </div>
            """.format(st.session_state.predicted_disease, st.session_state.confidence), 
            unsafe_allow_html=True)
        
        with result_col2:
            st.subheader("💡 Advisory")
            advisory = get_disease_advisory(st.session_state.predicted_disease)
            st.info(advisory)
    
    # Disclaimer
    st.divider()
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Disclaimer</h4>
        <p>
        <b>This system is for educational and demonstration purposes only.</b> 
        The predictions provided by this system should NOT be considered as medical advice, 
        diagnosis, or treatment recommendation. Always consult a qualified healthcare 
        professional for proper medical evaluation and treatment. This system is intended 
        to be used as a preliminary screening tool and educational demonstration of 
        machine learning in healthcare.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 0.9rem;">
        Disease Prediction System Using Symptoms | Machine Learning Research Project<br>
        Powered by Decision Tree Classifier & Streamlit
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
