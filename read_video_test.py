from collections import Counter

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import subprocess
import cv2
from deepface import DeepFace
import mediapipe as mp

video_data = st.file_uploader("Upload file", ['mp4', 'mov', 'avi'])

temp_file_to_save = './temp_file_1.mp4'
temp_file_result = './temp_file_2.mp4'

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
]


# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet.
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())



def detect_emotion(face_image):
    # Ensure the face image is not empty
    if face_image is None or face_image.size == 0:
        return None

    # Add a batch dimension to the face_image

    # Perform emotion detection using DeepFace
    emotions_list = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)

    # Check if the emotions_list is empty
    if not emotions_list:
        return None

    # Get the first result from the list
    emotions = emotions_list[0]

    emotion_label = emotions['dominant_emotion']
    return emotion_label

def mediapipe_face_detection(image):
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    results = mp_face_detection.process(image)

    if results.detections:
        ih, iw, _ = image.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = bbox

            # Ensure the bounding box coordinates are within image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(iw - x, w)
            h = min(ih - y, h)

            # Check if the bounding box has a valid size
            if w > 0 and h > 0:
                face_image = image[y:y + h, x:x + w]

                # Convert face_image to BGR format if it is not already in BGR format
                if not isinstance(face_image, np.ndarray) or face_image.shape[-1] != 3:
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

                # Perform emotion detection using DeepFace
                emotion_label = detect_emotion(face_image)
                current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if emotion_label is not None:
                    # Draw the emotion label on the frame
                    cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
if video_data:
    # save uploaded video to disc
    write_bytesio_to_file(temp_file_to_save, video_data)
    # read it with cv2.VideoCapture(),
    # so now we can process it with OpenCV functions
    cap = cv2.VideoCapture(temp_file_to_save)
    # grab some parameters of video to use them for writing a new, processed video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int
    st.write(width, height, frame_fps)
    # specify a writer to write a processed video to a disk frame by frame
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height), isColor=False)
    emotions_list = []
    timestamps = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion_dominant = mediapipe_face_detection(frame)
        emotion_label = detect_emotion(frame)
        # Draw the emotion label on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  ##<< Generates a grayscale (thus only one 2d-array)
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if emotion_label is not None:
            emotions_list.append(emotion_label)
            timestamps.append(current_timestamp)

        # Write the frame to the output video
        out_mp4.write(gray)
        # Create the Plotly timeline




        # Show the processed frame (you can comment this out for faster processing)
        # cv2.imshow('Emotion Detection', frame)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=timestamps, y=emotions_list, mode='markers+lines', name='Emotions', line=dict(color='blue', width=2)))

    fig.update_layout(
        title="Emotion Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Emotion",
        template="plotly_white"
    )

    # Compute the count of each emotion from the emotions_list
    emotion_counts = dict(Counter(emotions_list))

    # Create the Plotly pie chart for emotion summary
    fig_pie = go.Figure(data=[go.Pie(labels=list(emotion_counts.keys()), values=list(emotion_counts.values()))])

    fig_pie.update_layout(
        title="Emotion Summary",
        template="plotly_white"
    )
    # Display the Plotly timeline in Streamlit
    st.plotly_chart(fig)
    st.plotly_chart(fig_pie)
    cap.release()
    out_mp4.release()

    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    convertedVideo = "./testh264.mp4"
    subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))

    ## Show results
    col1, col2 = st.columns(2)
    col1.header("Original Video")
    col1.video(temp_file_to_save)
    col2.header("Output from OpenCV (MPEG-4)")
    col2.video(temp_file_result)
    col2.header("After conversion to H264")
    col2.video(convertedVideo)

    ## Close video files
    out_mp4.release()
    cap.release()
