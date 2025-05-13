import asyncio
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pydicom
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as T
import zipfile
import cv2
from io import BytesIO

# Ensure the event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:  # No event loop is running
    asyncio.set_event_loop(asyncio.new_event_loop())

# Disable Streamlit's file watcher for PyTorch
# st.set_option('server.fileWatcherType', 'none')

# Load the model
@st.cache_resource
def load_model(model_path):
    model_eval = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model_eval.roi_heads.box_predictor.cls_score.in_features
    model_eval.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model_eval.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_eval.eval()
    return model_eval

# Updated Prediction Function
def predict_and_visualize(image, model):
    with torch.no_grad():
        model.eval()
        outputs = model(image)  # Model returns a list of dictionaries

    if len(outputs) > 0:
        boxes = outputs[0]["boxes"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()

        return boxes, labels, scores
    return None, None, None

# Login Interface
def login_interface():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "pass":
            st.success("Logged in successfully!")
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid username or password.")

# Sidebar for mode selection
def sidebar_mode_selection():
    st.sidebar.title("Mode Selection")
    mode = st.sidebar.radio("Choose a mode:", ["Single Prediction", "Multiple Predictions"])
    return mode

# Uploading and Prediction Interface
def upload_and_predict_interface():
    st.title("Model Prediction and Visualization")
    model = load_model("fasterrcnn_dicom_tumor.pth")
    st.success("Model loaded successfully!")
    uploaded_file = st.file_uploader("Upload an image or DICOM file", type=["jpg", "png", "jpeg", "dcm"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".dcm"):
                st.write("Processing DICOM file...")
                ds = pydicom.dcmread(uploaded_file)
                image = ds.pixel_array
                image = (image - image.min()) / (image.max() - image.min())  # Normalize
                image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
                image = Image.fromarray(image)  # Convert to PIL image
            else:
                st.write("Processing image file...")
                image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format

            # Display the uploaded image
            st.image(image, caption="Uploaded Image",use_container_width=True)

            # Apply necessary transformations
            transform = T.Compose([
                T.Resize((256, 256)),  # Resize to 256x256
                T.ToTensor(),          # Convert to tensor
            ])
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Predict and visualize
            if st.button("Predict"):
                st.write("Processing...")
                boxes, labels, scores = predict_and_visualize(image_tensor, model)
                if boxes is not None:
                    # Get original image dimensions
                    original_width, original_height = image.size  # PIL image dimensions
                    resized_width, resized_height = 256, 256  # Resized dimensions used for the model

                    # Calculate scaling factors
                    scale_x = original_width / resized_width
                    scale_y = original_height / resized_height

                    # Plot the image with bounding boxes
                    fig, ax = plt.subplots(1, figsize=(10, 6))
                    ax.imshow(image)
                    for box, label, score in zip(boxes, labels, scores):
                        if score > 0.5:  # Confidence threshold
                            # Scale bounding box coordinates back to original dimensions
                            x1, y1, x2, y2 = box
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y

                            # Draw the bounding box
                            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(x1, y1 - 10, f"Label: {label}, Score: {score:.2f}", color='red', fontsize=12, weight='bold')
                    st.pyplot(fig)
                else:
                    st.write("No objects detected.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Function to process multiple DICOM files
def process_multiple_dicom(zip_file, model):
    st.write("Processing ZIP file...")
    with zipfile.ZipFile(zip_file, "r") as z:
        dicom_files = [f for f in z.namelist() if f.endswith(".dcm")]
        if not dicom_files:
            st.error("No DICOM files found in the ZIP archive.")
            return

        frames = []
        for dicom_file in dicom_files:
            with z.open(dicom_file) as file:
                ds = pydicom.dcmread(file)
                image = ds.pixel_array
                image = (image - image.min()) / (image.max() - image.min())  # Normalize
                image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
                image = Image.fromarray(image)  # Convert to PIL image

                # Apply transformations
                transform = T.Compose([
                    T.Resize((256, 256)),  # Resize to 256x256
                    T.ToTensor(),          # Convert to tensor
                ])
                image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

                # Predict
                boxes, labels, scores = predict_and_visualize(image_tensor, model)
                if boxes is not None:
                    # Get original image dimensions
                    original_width, original_height = image.size
                    resized_width, resized_height = 256, 256

                    # Calculate scaling factors
                    scale_x = original_width / resized_width
                    scale_y = original_height / resized_height

                    # Draw bounding boxes
                    fig, ax = plt.subplots(1, figsize=(10, 6))
                    ax.imshow(image)
                    for box, label, score in zip(boxes, labels, scores):
                        if score > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y

                            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(x1, y1 - 10, f"Label: {label}, Score: {score:.2f}", color='red', fontsize=12, weight='bold')

                    # Save the frame as an image
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    frame = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
                    frames.append(frame)
                    plt.close(fig)

        # Create a video from the frames
        if frames:
            st.write("Generating video...")
            video_path = "output_video.avi"
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), 1, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

            # Display the video
            st.video(video_path)

# Main Function
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_interface()
    else:
        mode = sidebar_mode_selection()
        if mode == "Single Prediction":
            upload_and_predict_interface()
        elif mode == "Multiple Predictions":
            st.title("Multiple Predictions")
            model = load_model("fasterrcnn_dicom_tumor.pth")
            st.success("Model loaded successfully!")
            uploaded_zip = st.file_uploader("Upload a ZIP file containing DICOM files", type=["zip"])
            if uploaded_zip:
                process_multiple_dicom(uploaded_zip, model)

if __name__ == "__main__":
    main()