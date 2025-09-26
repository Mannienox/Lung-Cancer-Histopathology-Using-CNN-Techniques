import streamlit as st
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Lung Disease Detector",
    page_icon="🧑‍⚕️",
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


# --- Configuration ---
MODEL_PATH = 'Model/resnet50_7_epochs_adam_lr_0_001.pth'
LABELS = ['lung_aca', 'lung_n', 'lung_scc']
NUM_CLASSES = len(LABELS)
device = "cpu"


# --- Model Loading and Caching ---
@st.cache_resource
def load_and_prepare_model(model_path, num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = False
    

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                        nn.Linear(num_ftrs, 512),       # First linear layer to a hidden size of 512
                        nn.ReLU(),                         # Non-linear activation function
                        nn.Dropout(p=0.2),                 # Dropout for regularization to prevent overfitting
                        nn.Linear(512, len(LABELS))        # Final linear layer to output the number of classes
                ).to(device)
    
    for param in model.layer4.parameters():
        param.requires_grad = True

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


st.markdown("<h1 style='text-align: center;'> Lung Disease Predictor </h1>", unsafe_allow_html=True)
#st.markdown("<p style='text-align: center;'>Upload a lung scan to get instant disease diagnosis.</p>", unsafe_allow_html=True)


# --- Sidebar Content ---
st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1.  **Upload Image:** Use the file uploader below to select a `.png`, `.jpg`, or `.jpeg` image of a maize leaf.
    2.  **View Image:** Your uploaded image will appear in the main area.
    3.  **Predict:** Click the `Diagnose` button to get the diagnosis.
    4.  **Results:** The prediction and confidence will be displayed.
    """
)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Histopathology Image Scan", type=["png", "jpg", "jpeg"])


# --- Main Area Content ---
if uploaded_file is not None:
    # Display the uploaded image in the main area
    st.image(uploaded_file, caption='', use_container_width=True)

    # Centralized Predict Button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button(label='Diagnose', use_container_width=True)

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

        if current_pred == 'lung_n':
            st.success(f"**Diagnosis: 🧑‍⚕️ {current_pred}**")
            st.info(f"**Confidence: {str(round(current_prob.item() * 100)) + "%"}**")
            st.markdown("---")
            st.markdown("Great news! Your Scan appears healthy.")
        else:
            st.error(f"**Diagnosis: 🚨 {current_pred}**")
            st.info(f"**Confidence: {str(round(current_prob * 100)) + "%"}**")
            st.warning("Immediate action is be required. Please consult histopathological experts treatment plans and recommendations.")
            st.markdown("---")

else:
    st.info("Please upload a scan to get a diagnosis.")
    # Reset session state if no image is uploaded
    st.session_state.prediction_made = False
    st.session_state.gemini_clicked = False

