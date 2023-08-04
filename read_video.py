import streamlit as st
import cv2
from PIL import Image
from retinaface import RetinaFace
from deepface import DeepFace

uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 1  # display every frame


def detect_emotions(image):
    faces = RetinaFace.detect_faces(image)
    for _, face in faces.items():
        facial_area = face["facial_area"]
        landmarks = face["landmarks"]
        # highlight facial area
        cv2.rectangle(image, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 0, 0), 2)

        # highlight the landmarks
        for k, v in landmarks.items():
            cv2.circle(image, tuple(map(int, v)), 1, (0, 255, 0), cv2.FILLED)

        # Get facial area for emotion detection
        top, right, bottom, left = face["facial_area"]
        face_image = image[top:bottom, left:right]

        # Perform emotion detection using deepface
        emotion = DeepFace.analyze(face_image, actions=['emotion'],enforce_detection=False)

        # Display the emotion
        emotion_label = max(emotion['emotion'], key=emotion['emotion'].get)
        st.write(f"Emotion: {emotion_label}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), faces


if uploaded_video is not None:  # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read())  # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
                unsafe_allow_html=True)  # display file name

    vidcap = cv2.VideoCapture(vid)  # load video from disk
    cur_frame = 0
    success = True

    while success:
        success, frame = vidcap.read()  # get next frame from video
        if cur_frame % frame_skip == 0:  # only analyze every n=300 frames
            pil_img = Image.fromarray(frame)  # convert opencv frame (with type()==numpy) into PIL Image
            st.image(pil_img)
            _, _ = detect_emotions(frame)  # Detect emotions in the frame
        cur_frame += 1
