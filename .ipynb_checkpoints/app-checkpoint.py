import streamlit as st
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import io
import os 
import uuid # For generating unique IDs for the scan logs
from datetime import datetime
import time
import requests 
from io import BytesIO 
import json 
import shutil # For saving file objects

# --- Local File System Mock for Development (Your requested local storage) ---
LOCAL_HISTORY_FILE = 'scan_history.json'
# --- NEW: Folder where images will be saved ---
LOCAL_IMAGE_FOLDER = 'history'
UPLOADED_FILE_PLACEHOLDER = "https://placehold.co/600x400/CCCCCC/000000?text=Image+Not+Saved" # Retained for clarity on static samples

# Helper function to correctly define the to_dict method for MockDoc objects
def _create_to_dict_method(data_dict):
    """Ensures the to_dict method correctly closes over and returns its specific data dictionary."""
    # Must accept 'self' even if it's not used, to satisfy Python method convention
    def to_dict_impl(self):
        return data_dict
    return to_dict_impl


class LocalFileStore:
    """A mock Firestore class that uses a local JSON file for persistence."""
    
    def __init__(self):
        # Ensure the JSON history file exists
        if not os.path.exists(LOCAL_HISTORY_FILE):
            try:
                with open(LOCAL_HISTORY_FILE, 'w') as f:
                    json.dump([], f)
            except IOError as e:
                st.error(f"Cannot create local history file: {e}")
                
        # --- NEW: Ensure the local image storage folder exists ---
        if not os.path.exists(LOCAL_IMAGE_FOLDER):
            try:
                os.makedirs(LOCAL_IMAGE_FOLDER, exist_ok=True)
            except IOError as e:
                st.error(f"Cannot create local image folder '{LOCAL_IMAGE_FOLDER}': {e}")


    def collection(self, *args):
        # We don't need real collections/documents for this simple local persistence
        return self
    
    def document(self, *args):
        return self
    
    def set(self, data):
        """Saves a new scan log entry to the local JSON file."""
        try:
            # 1. Read existing data
            with open(LOCAL_HISTORY_FILE, 'r') as f:
                history = json.load(f)
            
            # 2. Add new data (simulating a document structure for consistency)
            new_entry = {
                'id': str(uuid.uuid4()),
                'predicted_class': data.get('predicted_class'),
                'probability': data.get('probability'),
                'image_url': data.get('image_url'), # Now stores the local path or remote URL
                'timestamp': data.get('timestamp').isoformat() # Convert datetime to ISO string for JSON
            }
            history.append(new_entry)
            
            # 3. Write back the updated data
            with open(LOCAL_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=4)
                
            return new_entry # Return the entry for success message
        except Exception as e:
            st.error(f"Local save failed: {e}")
            return None
        
    def stream(self):
        """Streams the content of the local JSON file (most recent first)."""
        try:
            with open(LOCAL_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                
            # Convert JSON data back to a "document" structure for consistency with Firestore logic
            # Sort by timestamp (most recent first)
            history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            mock_docs = []
            for item in history:
                # Mock object structure to mimic firestore doc.to_dict()
                timestamp_obj = datetime.fromisoformat(item.get('timestamp'))
                mock_doc_data = {
                    'predicted_class': item.get('predicted_class'),
                    'probability': item.get('probability'),
                    'image_url': item.get('image_url'), # Retrieve the image URL/Path
                    'timestamp': timestamp_obj
                }
                
                # Using the helper function to avoid potential lambda closure issues
                mock_docs.append(type('MockDoc', (object,), {
                    'id': item['id'],
                    'to_dict': _create_to_dict_method(mock_doc_data),
                    'exists': True
                })())
            return mock_docs

        except (FileNotFoundError, json.JSONDecodeError):
            return []
        except Exception as e:
            st.error(f"Local load failed: {e}")
            return []

# --- Determine Environment and Init Persistence Store ---

# Use environment variable to check if we are in the hosted (Canvas/Streamlit Cloud) environment
IS_HOSTED_ENV = os.environ.get('__app_id') is not None

if IS_HOSTED_ENV:
    try:
        from firebase_admin import initialize_app, firestore, credentials
        # NOTE: The implementation of Firestore needs to handle file uploads to Storage.
        # Since we cannot implement Firebase Storage without explicit access, we will default
        # to the LocalFileStore if not running in the Canvas environment where Storage is
        # typically configured.
        firestore = LocalFileStore() # For simplicity in this back-and-forth, we stick to local mock
        st.warning("⚠️ **Running in hosted mock mode:** Persistent history uses local file system (`scan_history.json`). Full Firebase integration (including Storage for images) would be required for production.")

    except (ImportError, NameError):
        firestore = LocalFileStore()
        st.warning("⚠️ **Running in local mode:** Persistent history uses local file system (`scan_history.json`). This will NOT work on cloud platforms like Streamlit Cloud or Canvas.")

else:
    # Local development fallback when __app_id is missing
    firestore = LocalFileStore()
    st.warning("⚠️ **Running in local mode:** Persistent history uses local file system (`scan_history.json`). This will NOT work on cloud platforms like Streamlit Cloud or Canvas.")
    
    
# Define paths relative to the app ID and user ID (MANDATORY for Canvas)
appId = os.environ.get('__app_id', 'default-lung-detector')
userId = os.environ.get('__user_id', 'mock-user') 

# --- Configuration ---
MODEL_PATH = 'Model/resnet50_7_epochs_adam_lr_0_001.pth'
# LABELS: lung_aca = Adenocarcinoma, lung_n = Normal/Benign, lung_scc = Squamous Cell Carcinoma
LABELS = ['lung_aca', 'lung_n', 'lung_scc'] 
NUM_CLASSES = len(LABELS)
device = "cpu"

# --- Static Repository Images (MOCK DATA) ---
FIREBASE_URL_PLACEHOLDER = "https://your_firebase_storage_bucket.appspot.com/v0/b/..."
REPO_IMAGES = {
    "Adenocarcinoma (Sample 1)": f"{FIREBASE_URL_PLACEHOLDER}/artifacts/{appId}/sample_scans/lung_aca_sample_1.jpg",
    "Normal Tissue (Sample 1)": f"{FIREBASE_URL_PLACEHOLDER}/artifacts/{appId}/sample_scans/lung_n_sample_1.jpg",
    "Squamous Cell Carcinoma (Sample 1)": f"{FIREBASE_URL_PLACEHOLDER}/artifacts/{appId}/sample_scans/lung_scc_sample_1.jpg",
}
REPO_IMAGE_DISPLAY_NAMES = list(REPO_IMAGES.keys())


# --- Define Transforms (FIXED to match training) ---
def get_inference_transform():
    """Returns the standardized transform pipeline for inference."""
    return transforms.Compose([
        transforms.Resize(size=(64, 64)), 
        transforms.ToTensor()          
    ])

inference_transform = get_inference_transform()

# --- Model Loading and Caching ---
@st.cache_resource
def load_and_prepare_model(model_path, num_classes):
    """Loads and prepares the ResNet-50 model with modified final layer."""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)

    # 1. Freeze initial parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Define the new head (MUST EXACTLY MATCH TRAINING HEAD)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, len(LABELS))
    ).to(device)
    
    # 3. Ensure the unfreezing matches the training code
    for param in model.layer4.parameters():
        param.requires_grad = True

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}. Cannot perform prediction.")
        return None 
    except RuntimeError as e:
        st.error(f"Error loading model weights. Check if the model architecture matches the saved state_dict. Error: {e}")
        return None
        
    model.eval()
    return model

model = load_and_prepare_model(MODEL_PATH, NUM_CLASSES)


# --- NEW: Function to save the uploaded file to the local history folder ---
def save_uploaded_file_locally(uploaded_file):
    """Saves the contents of a st.file_uploader object to a local file."""
    try:
        # Generate a unique filename using a UUID and the original extension
        file_ext = uploaded_file.name.split('.')[-1]
        unique_filename = f"scan_{uuid.uuid4().hex}.{file_ext}"
        save_path = os.path.join(LOCAL_IMAGE_FOLDER, unique_filename)
        
        # Write the file content
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return save_path
        
    except Exception as e:
        st.error(f"Failed to save uploaded image locally: {e}")
        return None


# --- Data Functions (Unified Logic) ---

@st.cache_data(ttl=60) # Cache the history for 60 seconds
def get_recent_scans():
    """Fetches the user's most recent scan logs from persistence layer (Firestore or Local JSON)."""
    
    # Since we are using LocalFileStore for image persistence, the logic remains simple
    # to maintain consistency with the local image saving approach.
    
    scan_logs = []
    for doc in firestore.stream():
        data = doc.to_dict()
        if isinstance(data, dict):
             scan_logs.append({
                'id': doc.id,
                'predicted_class': data.get('predicted_class'),
                'probability': data.get('probability'),
                'image_url': data.get('image_url'), # Retrieve the image URL/Path
                # Convert Datetime object back to string for display
                'timestamp': data.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') 
            })
    return scan_logs

def save_scan_log(pred_class, probability, image_source_url):
    """Saves the result of the current prediction to persistence layer, including the image source URL."""
    
    # Use the path/url directly, no more placeholder logic needed for successful saves
    if image_source_url is None:
        st.error("Cannot save scan log: Image source URL is missing.")
        return
        
    data = {
        'predicted_class': pred_class,
        'probability': probability,
        'image_url': image_source_url, # Now stores the local path or remote URL
        'timestamp': datetime.now() # Use datetime object
    }
    
    # Logic for Local File System (Development Environment)
    if firestore.__class__.__name__ == 'LocalFileStore':
        result = firestore.set(data)
        if result:
            st.sidebar.success(f"Scan result saved to local history ({LOCAL_HISTORY_FILE})")
            get_recent_scans.clear() # Invalidate cache
        
    else:
        # Default fallback for hosted env (which currently also uses LocalFileStore mock)
        # In a real cloud setup, this would be the Firestore/Storage logic.
        st.error("Persistence layer configuration is ambiguous.")


# --- Prediction Function ---
def predict_single(model, image, transform, labels):
    """
    Performs prediction on a single image. If the image is an uploaded file,
    it saves it locally first and returns the local path.
    """
    
    pil_image = None
    image_path_or_url = None # To store the string path/URL used for saving the log
    
    if isinstance(image, str):
        # Case 1 & 2: Image is a URL (Repo) or local path (Dev testing)
        image_path_or_url = image 
        if image.startswith("http"):
            try:
                response = requests.get(image, timeout=10)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                st.error(f"Error fetching image from URL: {e}")
                return None, None, None, None
        
        elif os.path.exists(image):
            try:
                pil_image = Image.open(image).convert("RGB")
            except Exception:
                st.error(f"Error opening local image file: {image}")
                return None, None, None, None
        else:
             st.error(f"Error: The browsed image path/URL '{image}' was not valid or found.")
             return None, None, None, None
             
    # Case 3: Image is a file-like object (from st.file_uploader)
    else:
        # --- CRITICAL CHANGE: Save the uploaded file locally immediately ---
        local_path = save_uploaded_file_locally(image) 
        
        if local_path:
            image_path_or_url = local_path # Set the local path as the source for the log
            try:
                # Open from the saved local path to ensure consistency
                pil_image = Image.open(local_path).convert("RGB") 
            except Exception:
                st.error("Error opening uploaded image file.")
                return None, None, None, None
        else:
            return None, None, None, None
    
    # Check if image was successfully loaded
    if pil_image is None:
        return None, None, None, None

    # Apply the FIXED transforms
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)

    predicted_prob, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = labels[predicted_idx.item()]
    
    # Get probabilities for all classes
    all_probs = probabilities.squeeze().tolist()

    return predicted_class, predicted_prob.item(), all_probs, image_path_or_url

# --- Callback Function for History Click ---
def set_history_display_mode(image_url, predicted_class, timestamp_short):
    """
    Updates session state to display a history image and forces the radio button mode.
    This is used as the 'on_click' callback for history buttons.
    """
    # Now, image_url is either a remote URL or a local path (string)
    if image_url: 
        st.session_state.image_to_display = image_url
        st.session_state.image_display_caption = f"Previous Scan: {predicted_class.upper()} ({timestamp_short})"
        st.session_state.image_source_mode = "View Recent Scans"
        st.session_state.prediction_made = False
    else:
        # This should only happen if a save failed or the path/URL was genuinely corrupted
        st.error("The image path or URL for this scan is missing or invalid.")

# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Lung Disease Detector",
    page_icon="🧑‍⚕️",
    layout="centered"
)

# --- Initialize session state ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'pred_result' not in st.session_state: 
    st.session_state.pred_result = None
if 'prob_result' not in st.session_state:
    st.session_state.prob_result = None
if 'all_probs' not in st.session_state:
    st.session_state.all_probs = None
if 'results_expanded' not in st.session_state:
    st.session_state.results_expanded = False
if 'image_to_display' not in st.session_state: # State to hold the image URL/Path for the main viewer
    st.session_state.image_to_display = None
if 'image_display_caption' not in st.session_state: # State to hold the image caption
    st.session_state.image_display_caption = None
# State to persist the radio button selection
if 'image_source_mode' not in st.session_state:
    st.session_state.image_source_mode = "Upload New Scan"


st.markdown("<h1 style='text-align: center;'> Lung Histopathology Predictor </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a high-resolution microscopic scan for classification into Adenocarcinoma (ACA), Normal (N), or Squamous Cell Carcinoma (SCC).</p>", unsafe_allow_html=True)


# --- Sidebar Content ---
st.sidebar.header("Image Source")
# Use the session state variable for the radio button value and key
image_source_mode = st.sidebar.radio(
    "Select Input Mode:",
    ("Upload New Scan", "Browse Static Repo Samples", "View Recent Scans"),
    key='image_source_mode'
)

# Initialize source variables
uploaded_file = None
browsed_file_path = None
image_source = None # Unified source variable

if image_source_mode == "Upload New Scan":
    st.sidebar.markdown(
        """
        1.  **Upload Image:** Select a `.png`, `.jpg`, or `.jpeg` image scan.
        """
    )
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload Histopathology Image Scan", type=["png", "jpg", "jpeg"])
    image_source = uploaded_file
    # If the user is uploading a new image, reset the display to the new image
    if uploaded_file is not None:
         st.session_state.image_to_display = image_source
         st.session_state.image_display_caption = 'Input Histopathology Scan'

elif image_source_mode == "Browse Static Repo Samples":
    st.sidebar.markdown(
        """
        1.  **Browse:** Select an existing image from the repository for analysis.
        """
    )
    st.sidebar.markdown("---")
    selected_display_name = st.sidebar.selectbox(
        "Select a Sample Image:",
        options=["Select an image..."] + REPO_IMAGE_DISPLAY_NAMES
    )
    
    if selected_display_name != "Select an image...":
        browsed_file_path = REPO_IMAGES[selected_display_name]
        
        # Check if the URL placeholder is still there and warn the user
        if FIREBASE_URL_PLACEHOLDER in browsed_file_path:
             st.error("🚨 **Configuration Error:** Please update the `FIREBASE_URL_PLACEHOLDER` in the code to your actual Firebase Storage bucket URL to use sample images.")
             image_source = None
        else:
            image_source = browsed_file_path
            st.session_state.image_to_display = image_source
            st.session_state.image_display_caption = f'Repo Sample: {selected_display_name}'

    else:
        image_source = None
        st.session_state.image_to_display = None
        st.session_state.image_display_caption = None

elif image_source_mode == "View Recent Scans":
    # Reset prediction state when viewing history
    st.session_state.prediction_made = False 

    st.sidebar.markdown(
        """
        1.  **History:** Click on any scan entry to view the image used for that diagnosis.
        """
    )
    st.sidebar.markdown("---")
    recent_scans = get_recent_scans()
    
    if recent_scans:
        st.sidebar.subheader("Prediction History")
        
        # Display logs as clickable expanders
        for scan in recent_scans:
            result_color = "green" if scan['predicted_class'] == 'lung_n' else "red"
            timestamp_short = scan['timestamp'].split(' ')[0]
            
            with st.sidebar.expander(
                f"**{scan['predicted_class'].upper()}** | {timestamp_short}",
                expanded=False
            ):
                # Use st.markdown() inside the expander to still use color coding
                st.markdown(f"**Result:** <span style='color:{result_color}'>{scan['predicted_class'].upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{scan['probability']:.4f}`")
                st.markdown(f"**Date:** `{scan['timestamp']}`")
                
                # Button to load the image into the main viewer
                st.button(
                    "View Image", 
                    key=f"view_img_{scan['id']}",
                    on_click=set_history_display_mode,
                    args=[scan['image_url'], scan['predicted_class'], timestamp_short]
                )
                    
    else:
        st.sidebar.info("No recent scan history found for this user.")

st.sidebar.markdown("---")

# --- Main Area Content ---

# 1. Display the Image based on the current state
if st.session_state.image_to_display is not None:
    # Check if we are in an active prediction mode (not history viewing)
    if image_source_mode in ["Upload New Scan", "Browse Static Repo Samples"]:
        active_mode = True
        
        # If active mode and the source is the uploaded file object (before it's saved/diagnosed)
        if st.session_state.image_to_display == uploaded_file and uploaded_file is not None:
            display_source = uploaded_file # Display the file object
        else:
            # Display the path or URL string
            display_source = st.session_state.image_to_display

    else:
        active_mode = False
        display_source = st.session_state.image_to_display

    # Only show image if the source is not None
    if display_source:
        st.image(display_source, caption=st.session_state.image_display_caption, use_container_width=True)
    
else:
    active_mode = False
    display_source = None
    
# 2. Centralized Predict Button (Only visible if we have an image to process and we are in an active mode)
if image_source is not None and model is not None and active_mode:

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button(label='Diagnose', use_container_width=True)

    # 3. If predict button is clicked, perform prediction and store in session state
    if predict_button:
        # Reset state flags
        st.session_state.prediction_made = True
        st.session_state.results_expanded = True # Auto-open results on new diagnosis
        # Reset image_to_display caption to the default for a new prediction
        st.session_state.image_display_caption = 'Input Histopathology Scan'


        with st.spinner('Diagnosing image...'):
            # The prediction function now saves the file locally if it's an upload, 
            # and returns the local path or URL
            pred, prob, all_probs, image_path_or_url = predict_single(
                model=model, 
                image=image_source, 
                transform=inference_transform, 
                labels=LABELS
            )
            # Check for prediction failure
            if pred is None:
                st.session_state.prediction_made = False # Prevent results display if file not found
            else:
                st.session_state.pred_result = pred
                st.session_state.prob_result = prob
                st.session_state.all_probs = all_probs
                
                # --- Save the result, passing the new local path or original URL ---
                save_scan_log(pred, prob, image_path_or_url)
                
                # Update the main viewer's source to the permanent path/URL
                st.session_state.image_to_display = image_path_or_url


# 4. Always display results if a prediction has been made (using session_state)
if st.session_state.prediction_made:
    current_pred = st.session_state.pred_result
    current_prob = st.session_state.prob_result
    all_probs = st.session_state.all_probs
    
    st.markdown("---") # Separator for results

    # --- Display Primary Diagnosis ---
    if current_pred == 'lung_n':
        st.success(f"**Diagnosis: 🧑‍⚕️ {current_pred.upper()}** - Likely Normal Tissue")
        st.info("Great news! Your Scan appears healthy.")
    else:
        st.error(f"**Diagnosis: 🚨 {current_pred.upper()}** - Highly Suggestive of Cancer")
        st.warning("Immediate action may be required. Please consult histopathological experts for treatment plans and recommendations.")
        
    st.markdown("---")
    
    # --- Collapsible Analysis Results (The 'Logic' Button) ---
    with st.expander("View Analysis Results (Model Logic)", expanded=st.session_state.results_expanded):
        
        # Display Confidence
        st.info(f"**Confidence: {current_prob:.4f}** (Model certainty for this diagnosis)")
        
        st.markdown("---") 

        # Display all probabilities
        st.markdown("### All Class Probabilities")
        
        # Zip labels and probabilities, sort by probability (descending)
        sorted_probs = sorted(zip(LABELS, all_probs), key=lambda x: x[1], reverse=True)
        
        # Display as a structured list
        for label, prob in sorted_probs:
            # Highlight the predicted class
            display_label = label.replace('_', ' ').title() # Format label nicely
            if label == current_pred:
                st.markdown(f"**- {display_label}: `{prob:.8f}`** (Predicted)")
            else:
                st.markdown(f"- {display_label}: `{prob:.8f}`")
    
    st.markdown("---") 

elif image_source_mode not in ["View Recent Scans"] and st.session_state.image_to_display is None:
    # This is shown when the app first loads or when no image is selected/uploaded
    st.info("Please select an image source (Upload or Browse) and an image to get a diagnosis.")
    # Reset session state if no image is uploaded
    st.session_state.prediction_made = False


st.markdown("---")
st.caption("🚨 **Medical Disclaimer:** This tool provides a preliminary classification for informational purposes only. It is **NOT** a diagnostic tool. Always consult a qualified healthcare professional, such as a pathologist or oncologist, for a definitive diagnosis.")