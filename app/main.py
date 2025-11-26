import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load the class indices
class_indices_path = f"{working_dir}/class_indices.json"
class_indices = json.load(open(class_indices_path))
# Convert keys to integers
class_indices = {int(k): v for k, v in class_indices.items()}

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image using MobileNetV2 preprocessing
    CRITICAL: Must match the preprocessing used during training!
    """
    # Load the image and convert to RGB (handles RGBA, grayscale, etc.)
    img = Image.open(image_path).convert("RGB")
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Apply MobileNetV2 preprocessing (scales to [-1, 1])
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    """
    Predict the disease class and confidence of a crop image
    """
    # Preprocess the image
    preprocessed_img = load_and_preprocess_image(image_path)
    
    # Get predictions
    predictions = model.predict(preprocessed_img, verbose=0)
    
    # Get predicted class index and confidence
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    
    # Map to class name
    predicted_class_name = class_indices[predicted_class_index]
    
    return predicted_class_name, confidence

def format_disease_name(prediction):
    """Format the disease name for better readability"""
    parts = prediction.split('___')
    if len(parts) == 2:
        crop = parts[0].replace('_', ' ').title()
        disease = parts[1].replace('_', ' ').title()
        return f"**Crop:** {crop}\n\n**Disease:** {disease}"
    else:
        return prediction.replace('___', ' ').replace('_', ' ').title()


# ============================================================================
# STREAMLIT APP UI
# ============================================================================

# Page config
st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌾",
    layout="centered"
)

# Title and description
st.title('🌾 Crop Disease Classifier')
st.markdown('##### Using Deep Learning')
st.write('Upload an image of a crop leaf to identify diseases')

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **Crop Disease Classification Using Deep Learning**
    
    This application uses a Convolutional Neural Network (CNN) trained on 38 crop disease classes 
    to identify diseases from leaf images.
    """)
    
    st.markdown("---")
    
    st.header("🌾 Supported Crops")
    st.write("""
    - 🍎 Apple
    - 🌽 Corn (Maize)
    - 🍇 Grape
    - 🥔 Potato
    - 🍅 Tomato
    - 🫐 Blueberry
    - 🍒 Cherry
    - 🍑 Peach
    - 🌶️ Pepper
    - 🍓 Strawberry
    - And more...
    """)
    
    st.markdown("---")
    
    st.header("📊 Model Details")
    st.write(f"""
    - **Total Classes:** {len(class_indices)}
    - **Architecture:** MobileNetV2 (Transfer Learning)
    - **Model Accuracy:** 94.96%
    - **Dataset:** PlantVillage
    """)
    
    st.markdown("---")
    
    st.info("💡 **Tip:** For best results, upload clear images with good lighting and visible leaf symptoms.")

# File uploader
uploaded_image = st.file_uploader(
    "Choose a crop leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a crop leaf"
)

if uploaded_image is not None:
    # Create two columns for image and results
    col1, col2 = st.columns(2)
    
    with col1:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Crop Image", use_container_width=True)
    
    with col2:
        # Classify button
        if st.button('🔍 Classify Disease', type="primary", use_container_width=True):
            with st.spinner('🔄 Analyzing crop image...'):
                try:
                    # Get prediction
                    prediction, confidence = predict_image_class(
                        model, 
                        uploaded_image, 
                        class_indices
                    )
                    
                    # Display results
                    st.success('✅ Classification Complete!')
                    
                    # Format and display the prediction
                    st.markdown(f"### 🌿 Diagnosis:")
                    formatted_prediction = format_disease_name(prediction)
                    st.markdown(formatted_prediction)
                    
                    st.markdown(f"### 📊 Confidence Score:")
                    st.progress(confidence)
                    st.markdown(f"**{confidence*100:.2f}%**")
                    
                    # Confidence-based recommendations
                    if confidence >= 0.90:
                        st.success("✅ **High Confidence:** Very reliable diagnosis")
                    elif confidence >= 0.75:
                        st.info("ℹ️ **Good Confidence:** Reliable diagnosis")
                    elif confidence >= 0.60:
                        st.warning("⚠️ **Moderate Confidence:** Consider expert verification")
                    else:
                        st.error("❌ **Low Confidence:** Please upload a clearer image")
                    
                    # Action recommendation based on disease
                    st.markdown("---")
                    if "healthy" in prediction.lower():
                        st.success("🎉 **Good News!** The crop appears to be healthy.")
                    else:
                        st.warning("⚠️ **Action Required:** Disease detected. Consult agricultural experts for treatment.")
                    
                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
                    st.info("Please try uploading a different image.")

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Show metrics
if uploaded_image is None:
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric(
            label="🎯 Accuracy",
            value="94.96%"
        )
    
    with col_b:
        st.metric(
            label="🌾 Crops",
            value="14+"
        )
    
    with col_c:
        st.metric(
            label="🦠 Classes",
            value=f"{len(class_indices)}"
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <h4 style='color: #2E7D32; margin-bottom: 1rem;'>Crop Disease Classification Using Deep Learning</h4>
        <p>Powered by MobileNetV2 & TensorFlow</p>
        <p style='font-size: 0.9em; color: gray; margin-top: 1rem;'>
            Built with ❤️ using Streamlit | Dataset: PlantVillage
        </p>
    </div>
    """,
    unsafe_allow_html=True
)