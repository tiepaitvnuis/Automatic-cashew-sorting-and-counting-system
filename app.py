# Python In-built packages
from pathlib import Path
import PIL
import cv2

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page heading
st.title("Object Detection And Tracking cashew using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio("Select Task", ["Detection"])

confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == "Detection":
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

source_vid = None
# If image is selected
if source_radio == settings.IMAGE:
    source_vid = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_vid is None:
                default_vid_path = str(settings.DEFAULT_IMAGE)
                default_vid = PIL.Image.open(default_vid_path)
                st.image(
                    default_vid_path, caption="Default Image", use_column_width=True
                )
            else:
                uploaded_vid = PIL.Image.open(source_vid)
                st.image(source_vid, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_vid is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(
                default_detected_image_path,
                caption="Detected Image",
                use_column_width=True,
            )
        else:
            if st.sidebar.button("Detect Objects"):
                res = model.predict(uploaded_vid, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption="Detected Image", use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    source_vid = st.sidebar.file_uploader(
        "Choose a video...",
        type=(
            # A Quang xóa mục này ko phải t :)
        ),
    )

    col1, col2 = st.columns(2)
    default_vid_path = str(settings.DEFAULT_VIDEO)
    default_vid = open(default_vid_path, 'rb').read()

    with col1:
        try:
            if source_vid is None:
                st.video(default_vid)
            else:
                uploaded_vid = source_vid.read() # Đã fix lỗi upload video
                st.video(uploaded_vid)
        except Exception as ex:
            st.error("Error occurred while opening the video.")
            st.error(ex)
    with col2:
        if source_vid is None:
            default_detected_video_path = str(settings.DEFAULT_DETECT_VIDEO)
            default_detected_video = open(default_detected_video_path, 'rb').read()
            st.video(default_detected_video)
            # Note: sau một vài tìm hiểu xác định đc rằng trình phát video của Streamlit ko hỗ trợ file .avi
            # Đã sửa bằng cách convert sang file default .mp4
        else:
            if st.sidebar.button("Detect Objects"):
                res = model.predict(uploaded_vid, conf=confidence) # https://docs.ultralytics.com/modes/predict/
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.video(res_plotted)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No video is uploaded yet!")

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")