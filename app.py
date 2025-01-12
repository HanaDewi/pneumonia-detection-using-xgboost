import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io
import os

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Check if model exists in the current directory
        if not os.path.exists('model/densenet121_model_clahe_final.keras'):
            st.error("Model file not found. Please ensure the model is in the 'model' directory.")
            return None
        model = tf.keras.models.load_model('model/densenet121_model_clahe_final.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def apply_clahe(image):
    """Apply CLAHE to a grayscale image"""
    # Convert to grayscale if image is RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Apply CLAHE
    clahe_img = clahe.apply(image)
    
    # Convert back to RGB (3 channels)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    
    return clahe_img

def preprocess_image(upload):
    # Read the uploaded image
    image = Image.open(upload)
    
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Resize image
    image = cv2.resize(image, (224, 224))
    
    # Apply CLAHE
    processed_image = apply_clahe(image)
    
    # Normalize
    processed_image = processed_image / 255.0
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    return processed_image

def get_confidence_text(probability):
    """Generate confidence text based on probability"""
    confidence = probability if probability > 0.5 else (1 - probability)
    confidence_percent = confidence * 100
    
    if confidence_percent >= 90:
        confidence_level = "very high"
    elif confidence_percent >= 70:
        confidence_level = "high"
    elif confidence_percent >= 50:
        confidence_level = "moderate"
    else:
        confidence_level = "low"
    
    return f"with a {confidence_percent:.1f}% confidence ({confidence_level} confidence)"

def main():
    st.set_page_config(
        page_title="Pneumonia Detection",
        page_icon="🫁",
        layout="wide"
    )
    
    st.title("🫁 Pneumonia Detection from Chest X-Ray")
    st.write("Upload a chest X-ray image to detect pneumonia")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded X-ray image.', use_container_width=True)
        
        with col2:
            st.write("Analyzing image...")
            
            try:
                # Preprocess the image
                processed_image = preprocess_image(uploaded_file)
                
                # Load model and make prediction
                model = load_model()
                
                if model is not None:
                    prediction = model.predict(processed_image)
                    probability = prediction[0][0]
                    
                    # Get confidence text
                    confidence_text = get_confidence_text(probability)
                    
                    # Display prediction result with confidence
                    if probability > 0.5:
                        st.error(f"🚨 PNEUMONIA DETECTED {confidence_text}", icon="🚨")
                    else:
                        st.success(f"✅ NORMAL {confidence_text}", icon="✅")
                    
                    # Add confidence bar with label
                    st.write("Confidence Level:")
                    st.progress(float(probability if probability > 0.5 else 1 - probability))
                    
                    # Add explanation
                    st.info("""
                    📋 **Interpretation Guide**:
                    - Very high confidence: 90-100%
                    - High confidence: 70-89%
                    - Moderate confidence: 50-69%
                    - Low confidence: <50%
                    
                    ⚠️ This is a screening tool and should not replace professional medical diagnosis.
                    """)
                
            except Exception as e:
                st.error(f"Error occurred during prediction: {str(e)}")
                st.write("Please make sure you've uploaded a valid chest X-ray image.")

if __name__ == "__main__":
    main()