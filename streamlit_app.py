import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io # Required for handling bytes data from camera input

# --- Configuration ---
MODEL_PATH = 'best_model.h5'  
IMG_HEIGHT = 128             # <--- IMPORTANT: Replace with the height your model expects
IMG_WIDTH = 128               # <--- IMPORTANT: Replace with the width your model expects
CLASS_NAMES = ["Organic", "Recyclable"] # <--- IMPORTANT: Ensure order matches model output (e.g., 0 = Organic, 1 = Recyclable)

# --- Model Loading ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource # Use st.cache_resource for non-data objects like models
def load_keras_model(path):
    """Loads the pre-trained Keras model."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # You might want to stop the app if the model can't load
        st.stop()
        return None # Return None explicitly if stopping

# --- Image Preprocessing ---
def preprocess_image(image_data, target_size):
    """
    Preprocesses the image for model prediction.
    - Opens image data (from file upload or camera)
    - Converts to RGB (handles potential RGBA, grayscale)
    - Resizes
    - Converts to NumPy array
    - Normalizes pixel values (adjust if your model expects different normalization)
    - Adds batch dimension
    """
    try:
        image = Image.open(image_data)

        # Convert to RGB if needed (e.g., PNGs with transparency, grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size)
        image_array = np.array(image)

        # Normalize pixel values (common practice: scale to 0-1)
        # <--- IMPORTANT: Adjust this if your model was trained with different normalization!
        image_array = image_array / 255.0

        # Add batch dimension (model expects input shape like (1, height, width, channels))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("♻️ Organic vs. Recyclable Waste Classifier 🗑️")
st.write("Upload an image or use your camera to classify waste.")

# Load the model (cached)
model = load_keras_model(MODEL_PATH)

if model: # Only proceed if the model loaded successfully
    # --- Input Method Selection ---
    input_method = st.radio(
        "Choose input method:",
        ("Upload an image", "Use camera"),
        horizontal=True
    )

    img_file_buffer = None
    if input_method == "Upload an image":
        img_file_buffer = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    elif input_method == "Use camera":
        img_file_buffer = st.camera_input("Take a picture")

    # --- Process and Predict ---
    if img_file_buffer is not None:
        # Display the uploaded/captured image
        st.image(img_file_buffer, caption="Your Image", use_column_width=True)

        # Preprocess the image
        # For camera input, img_file_buffer is already file-like bytes
        # For file uploader, it's also file-like bytes
        processed_image = preprocess_image(img_file_buffer, (IMG_HEIGHT, IMG_WIDTH))

        if processed_image is not None:
            # Make prediction
            try:
                prediction = model.predict(processed_image)
                # st.write(prediction) # Optional: show raw prediction output for debugging

                # Get the class index with the highest probability
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = prediction[0][predicted_index] * 100 # Get confidence score

                # Display the result
                st.subheader("Prediction:")
                if predicted_class == "Organic":
                    st.success(f" composting bin The waste is **{predicted_class}** (Confidence: {confidence:.2f}%)")
                    st.info("This item can likely be composted.")
                else: # Recyclable
                    st.info(f" recycling bin The waste is **{predicted_class}** (Confidence: {confidence:.2f}%)")
                    st.info("This item can likely be recycled. Please check local recycling guidelines.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.info("Please provide an image using one of the methods above.")
else:
    st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure the file exists and is valid.")

st.markdown("---")
st.write("Developed for Waste Management Project")