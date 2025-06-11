import streamlit as st
from PIL import Image
from predict_tf import predict_image

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:", layout="centered")
st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Upload or click a product image to see if it’s classified as Good or Abnormal using our AI model."
)

# Sidebar section
with st.sidebar:
    try:
        img = Image.open("./docs/overview_dataset.jpg")
        st.image(img)
    except Exception:
        st.warning("Overview image not found.")

    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is an AI-powered application that helps businesses streamline quality inspections. "
        "It ensures high standards by detecting scratches, dents, discolorations, and more on leather products."
    )

# Load image function
def load_uploaded_image(file):
    img = Image.open(file)
    return img

# Image input options
st.subheader("Select Image Input Method")
input_method = st.radio("Choose input method:", ["File Uploader", "Camera Input"])

uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow camera access.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Prediction section
submit = st.button(label="Submit a Product Image")
if submit:
    st.subheader("Prediction Output")
    img_file = uploaded_file_img if input_method == "File Uploader" else camera_file_img
    if img_file is None:
        st.error("No image provided.")
    else:
        with st.spinner(text="Analyzing the product..."):
            try:
                prediction = predict_image(img_file)
                st.success(prediction)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
