import streamlit as st
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import google.generativeai as genai


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Maize Disease Detector",
    page_icon="🌽",
    layout="centered"
)

# --- Initialize session state ---
# This ensures these variables exist from the start of the app's life
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'pred_result' not in st.session_state:
    st.session_state.pred_result = None
if 'prob_result' not in st.session_state:
    st.session_state.prob_result = None
if 'gemini_clicked' not in st.session_state: # To control Gemini recommendation display
    st.session_state.gemini_clicked = False


# --- Configuration ---
MODEL_PATH = 'Model/resnet50_10_epochs_adam_0_001.pth'
LABELS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
NUM_CLASSES = len(LABELS)
device = "cpu"

# --- Configure Gemini API (using Streamlit secrets) ---
gemini_model = None
try:
    gemini_api_key = "AIzaSyCtDAZOChhirBWa1GZ62vxedmGteMeOh_A"
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    pass


# --- Model Loading and Caching ---
@st.cache_resource
def load_and_prepare_model(model_path, num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False), # Removed inplace=True
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_and_prepare_model(MODEL_PATH, NUM_CLASSES)


# --- Prediction Function ---
def predict_single(model, image, transform, labels):
    pil_image = Image.open(image).convert("RGB")
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)

    predicted_prob, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = labels[predicted_idx.item()]

    return predicted_class, predicted_prob.item()


st.markdown("<h1 style='text-align: center;'>🌽 Maize Disease Predictor 🌽</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a maize leaf image and get an instant disease diagnosis.</p>", unsafe_allow_html=True)


# --- Sidebar Content ---
st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1.  **Upload Image:** Use the file uploader below to select a `.png`, `.jpg`, or `.jpeg` image of a maize leaf.
    2.  **View Image:** Your uploaded image will appear in the main area.
    3.  **Predict:** Click the 'Predict Now' button to get the diagnosis.
    4.  **Results:** The prediction and confidence will be displayed, along with a health status alert and treatment recommendations.
    """
)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Maize Leaf Image", type=["png", "jpg", "jpeg"])


# --- Main Area Content ---
if uploaded_file is not None:
    # Display the uploaded image in the main area
    st.image(uploaded_file, caption='Uploaded Maize Leaf Image', use_container_width=True)

    # Centralized Predict Button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button(label='Predict Now', use_container_width=True)

    # If predict button is clicked, perform prediction and store in session state
    if predict_button:
        st.session_state.prediction_made = True
        st.session_state.gemini_clicked = False # Reset Gemini recommendations when new prediction is made
        with st.spinner('Diagnosing image...'):
            pred, prob = predict_single(model=model, image=uploaded_file, transform=ResNet50_Weights.DEFAULT.transforms(), labels=LABELS)
            st.session_state.pred_result = pred
            st.session_state.prob_result = prob

    # Always display results if a prediction has been made (using session_state)
    if st.session_state.prediction_made:
        current_pred = st.session_state.pred_result
        current_prob = st.session_state.prob_result

        st.markdown("---") # Separator for results

        if current_pred == 'Healthy':
            st.success(f"**Diagnosis: 🌱 {current_pred}**")
            st.info(f"**Confidence: {current_prob:.4f}**")
            st.markdown("---")
            st.markdown("Great news! Your maize leaf appears healthy. Continue with good agricultural practices.")
        else:
            st.error(f"**Diagnosis: 🚨 {current_pred}**")
            st.info(f"**Confidence: {current_prob:.4f}**")
            st.warning("Immediate action might be required. Please consult local agricultural experts for specific treatment plans and recommendations.")
            st.markdown("---")

            if gemini_model: # Only show button if Gemini model was successfully configured
                st.subheader("🤖 AI-Powered Treatment Recommendations")
                recommend_button = st.button("Click to see AI-Powered Recommendation")

                # If recommend button is clicked, set session state flag
                if recommend_button:
                    st.session_state.gemini_clicked = True

                # Display recommendations if button was clicked previously
                if st.session_state.gemini_clicked:
                    prompt = f"""
                    You are an agricultural expert specializing in maize crops.
                    The diagnosed disease is: '{current_pred}'.

                    Provide detailed, practical, and comprehensive treatment recommendations for this maize disease.
                    Include:
                    1.  **Recommended actions for farmers:** What steps should they take immediately?
                    2.  **Possible control measures:** Are there cultural practices (e.g., crop rotation, residue management), resistant varieties, or biological controls?
                    3.  **Chemical treatments (if applicable):** Mention types of fungicides/pesticides if commonly used, but always advise consulting local experts for specific product names and dosages. Prioritize general categories.
                    4.  **Preventive measures:** How can future outbreaks be reduced?
                    5.  **Advice on when to consult local experts.**

                    Present the information clearly with bullet points where appropriate.
                    """
                    try:
                        with st.spinner("Generating detailed recommendations..."):
                            response = gemini_model.generate_content(prompt, stream=False)
                            treatment_recommendation = response.text
                            st.markdown(treatment_recommendation)
                    except Exception as e:
                        st.error(f"Failed to generate AI recommendations: {e}. Please try again later.")
                st.markdown("---") # Add separator after recommendations section
                st.caption("Disclaimer: These recommendations are for informational purposes only and should not replace professional agricultural advice. Always consult local agricultural experts for specific guidance.")
            else:
                st.info("AI recommendations are unavailable due to an API configuration issue.")

else:
    st.info("Please upload an image of a maize leaf to get a diagnosis.")
    # Reset session state if no image is uploaded
    st.session_state.prediction_made = False
    st.session_state.gemini_clicked = False


st.markdown("---")
st.caption("A capstone project by **Arewa Data Science Academy** Fellow. Learn more about this project on [GitHub](https://github.com/Mannienox/Maize_Disease_Prediction_Using_Transfer_Learning_Techniques/tree/main)")