import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
import joblib
from sklearn.metrics import roc_auc_score
from keras.applications.densenet import preprocess_input

# === CACHED MODEL LOADING ===
@st.cache_resource
def load_models():
    try:
        fe_model_path = "model/densenet_feature_extractor.keras"
        xgb_model_path = "model/model_xgb.pkl"

        if not os.path.exists(fe_model_path):
            st.error("❌ Feature extractor model not found.")
            return None, None
        if not os.path.exists(xgb_model_path):
            st.error("❌ xgb classifier model not found.")
            return None, None

        fe_model = tf.keras.models.load_model(fe_model_path, compile=False)
        xgb_model = joblib.load(xgb_model_path)
        return fe_model, xgb_model
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# === HEURISTIC CHECK FOR NON-XRAY ===
def is_probably_xray(image_np):
    """Heuristic check: Returns False if image has too many saturated colors."""
    if len(image_np.shape) == 3:
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[..., 1]
        high_sat_pixels = np.sum(saturation > 60)
        total_pixels = saturation.size
        ratio = high_sat_pixels / total_pixels
        return ratio < 0.05  # Less than 5% of pixels are high saturation
    return True  # Grayscale is fine

# === CLAHE PREPROCESSING ===
def apply_clahe(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(image)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_img

def preprocess_image(upload):
    image = Image.open(upload)
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    clahe_img = apply_clahe(image)
    clahe_display = clahe_img.copy()
    processed_image = clahe_img / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image, clahe_display

def get_confidence_text(probability):
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
    
    st.markdown(
            """
            <div style='text-align: center;'>
                <h1 style='margin-bottom: 0.3em;'>🫁 Pneumonia Detection from Chest X-Ray</h1>
                <h3>Welcome to Pneumonia Detection System</h3>
                <p>
                This AI tool helps analyze chest X-ray images for preliminary pneumonia detection.<br>
                Upload your chest X-ray image below to get started.<br><br>
                <b style='color: red;'>Important:</b> This is for educational purposes only. Always consult a doctor for proper medical diagnosis and treatment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )



    
    with st.expander("ℹ️ What kind of images should I upload?"):
        st.write("""
        **This tool is designed for chest X-ray images only:**
        - Frontal chest X-ray
        - Clear, good quality medical X-ray images
        - Grayscale X-ray images
        
        **Not suitable for:**
        - Regular photos or selfies
        - CT scans or MRI images
        - X-rays of other body parts
        - Low quality or blurry images
        """)

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image_check = Image.open(uploaded_file)
            image_np = np.array(image_check)

            if not is_probably_xray(image_np):
                st.warning("⚠️ Gambar yang diunggah kemungkinan bukan X-ray dada.")
                proceed = st.checkbox("Lanjutkan meskipun gambar kemungkinan bukan X-ray?")
                if not proceed:
                    st.info("Jika tidak, silakan unggah gambar X-ray lain untuk melanjutkan.")
                    st.stop()

            processed_image, clahe_display = preprocess_image(uploaded_file)

            fe_model, xgb_model = load_models()

            if fe_model is not None and xgb_model is not None:
                features = fe_model.predict(processed_image)
                features = features.reshape(1, -1)
                probability = xgb_model.predict_proba(features)[0][1]

                confidence_text = get_confidence_text(probability)
                label_text = "🚨 PNEUMONIA DETECTED" if probability > 0.5 else "✅ NORMAL"
                label_color = st.error if probability > 0.5 else st.success

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(uploaded_file, caption="📷 Original X-ray", use_container_width=True)
                with img_col2:
                    st.image(clahe_display, caption="⚙️ After CLAHE", use_container_width=True)

                label_color(f"{label_text} {confidence_text}")
                st.write("Confidence Level:")
                st.progress(float(probability if probability > 0.5 else 1 - probability))

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
