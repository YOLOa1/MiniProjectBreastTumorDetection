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
import os
import tempfile
from io import BytesIO


st.set_page_config(page_title="Tumor Detection Demo", page_icon="🩺")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

@st.cache_resource
def load_model(model_path):
    model_eval = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model_eval.roi_heads.box_predictor.cls_score.in_features
    model_eval.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model_eval.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_eval.eval()
    return model_eval

def predict_and_visualize(image, model):
    with torch.no_grad():
        model.eval()
        outputs = model(image)
    if len(outputs) > 0:
        boxes = outputs[0]["boxes"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        return boxes, labels, scores
    return None, None, None

def Home_page():
    st.markdown("# 🩺 Tumor Detection App")
    st.write(
        """
        This app allows you to detect the tumor's zone in medical images
        ## Instructions:
        1. **Single Prediction**: Upload a single image(JPG,PNG,JPEG) or DICOM file to detect tumors.
        2. **Multiple Predictions**: Upload a ZIP file containing multiple Images or DICOM files to detect tumors in all images, the output is a video.

        ### Notes: 
        **Model**:  The model used is a Faster R-CNN trained on DICOM images for tumor detection.\\
        **Performance**:  The model may take some time to process images, especially for larger files.\\
        **Output**:  The output will show the detected tumor zones with bounding boxes and labels.\\
        **Medical Images**:  Axial CT images are recommended for better results.
        

        """
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Single Prediction"):
            st.session_state["page"] = "Single Prediction"
    with col2:
        if st.button("Go to Multiple Predictions"):
            st.session_state["page"] = "Multiple Predictions"

def single_prediction_page():
    st.markdown("# Single Prediction")
    model = load_model("fasterrcnn_dicom_tumor.pth")
    st.success("Model loaded successfully!")
    uploaded_file = st.file_uploader("Upload an image or DICOM file", type=["jpg", "png", "jpeg", "dcm"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".dcm"):
                st.write("Processing DICOM file...")
                ds = pydicom.dcmread(uploaded_file)
                image = ds.pixel_array
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            else:
                st.write("Processing image file...")
                image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0)
            if st.button("Predict"):
                st.write("Processing...")
                boxes, labels, scores = predict_and_visualize(image_tensor, model)
                if boxes is not None:
                    original_width, original_height = image.size
                    resized_width, resized_height = 256, 256
                    scale_x = original_width / resized_width
                    scale_y = original_height / resized_height
                    fig, ax = plt.subplots(1, figsize=(10, 6))
                    ax.imshow(image,cmap="gray")
                    for box, label, score in zip(boxes, labels, scores):
                        if score > 0.5:
                            x1, y1, x2, y2 = box
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y
                            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(x1, y1 - 10, f"Label: {label}, Score: {score:.2f}", color='red', fontsize=12, weight='bold')
                    st.pyplot(fig)
                else:
                    st.write("No objects detected.")
        except Exception as e:
            st.error(f"Error processing file: {e}")


def process_multiple_dicom(zip_file, model):
    st.write("Processing ZIP file...")

    # Use session_state to cache processed images for the current ZIP file
    if "last_zip_name" not in st.session_state or st.session_state.get("last_zip_name") != zip_file.name:
        with zipfile.ZipFile(zip_file, "r") as z:
            dicom_files = sorted([f for f in z.namelist() if f.endswith(".dcm")])
            if not dicom_files:
                st.error("No DICOM files found in the ZIP archive.")
                return

            images = []
            image_names = []

            for idx, dicom_file in enumerate(dicom_files):
                st.write(f"Processing slice {idx+1}/{len(dicom_files)}: {dicom_file}")
                with z.open(dicom_file) as file:
                    ds = pydicom.dcmread(file)
                    image = ds.pixel_array
                    image = (image - image.min()) / (image.max() - image.min())
                    image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)

                    transform = T.Compose([
                        T.Resize((256, 256)),
                        T.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0)
                    boxes, labels, scores = predict_and_visualize(image_tensor, model)

                    fig, ax = plt.subplots(1, figsize=(6, 6))
                    ax.imshow(image, cmap="gray")
                    if boxes is not None:
                        original_width, original_height = image.size
                        scale_x = original_width / 256
                        scale_y = original_height / 256
                        for box, label, score in zip(boxes, labels, scores):
                            if score > 0.5:
                                x1, y1, x2, y2 = box
                                x1 *= scale_x
                                y1 *= scale_y
                                x2 *= scale_x
                                y2 *= scale_y
                                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                        linewidth=2, edgecolor='r', facecolor='none')
                                ax.add_patch(rect)
                                ax.text(x1, y1 - 10, f"Label: {label}, Score: {score:.2f}",
                                        color='red', fontsize=10, weight='bold')
                    ax.axis("off")

                    buf = BytesIO()
                    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    buf.seek(0)
                    images.append(buf.read())
                    image_names.append(f"slice_{idx+1}.png")

            # Save results in session_state
            st.session_state["multi_pred_images"] = images
            st.session_state["multi_pred_names"] = image_names
            st.session_state["last_zip_name"] = zip_file.name
    else:
        images = st.session_state["multi_pred_images"]
        image_names = st.session_state["multi_pred_names"]

    # Display with slider and download button
    if images:
        idx = st.slider("Select slice", 0, len(images) - 1, 0)
        st.image(images[idx], caption=f"Slice {idx+1} of {len(images)}", use_container_width=True)

        # Download all images as zip
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file_out:
            for img_bytes, img_name in zip(images, image_names):
                zip_file_out.writestr(img_name, img_bytes)
        zip_buffer.seek(0)
        st.download_button(
            label="Download all slices as ZIP",
            data=zip_buffer,
            file_name="predicted_slices.zip",
            mime="application/zip"
        )

def multiple_predictions_page():
    st.markdown("# Multiple Predictions")
    model = load_model("fasterrcnn_dicom_tumor.pth")
    st.success("Model loaded successfully!")
    uploaded_zip = st.file_uploader("Upload a ZIP file containing DICOM files", type=["zip"])
    if uploaded_zip:
        process_multiple_dicom(uploaded_zip, model)
def login_page():
    st.markdown("# 🔐 Login")
    for key in st.session_state.keys():
        if key.startswith("sidebar_"):
            del st.session_state[key]
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "pass":
            st.success("Logged in successfully!")
            st.session_state["logged_in"] = True
            st.session_state["page"] = "Home"
            st.rerun()
        else:
            st.error("Invalid username or password.")

def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    if not st.session_state["logged_in"]:
        st.session_state["page"] = "Home"
        login_page()
    else:
        # Sidebar navigation (only after login)
        st.sidebar.header("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ("Home", "Single Prediction", "Multiple Predictions"),
            index=["Home", "Single Prediction", "Multiple Predictions"].index(st.session_state["page"])
        )
        st.session_state["page"] = page

        if page == "Home":
            Home_page()
        elif page == "Single Prediction":
            single_prediction_page()
        elif page == "Multiple Predictions":
            multiple_predictions_page()

if __name__ == "__main__":
    main()
