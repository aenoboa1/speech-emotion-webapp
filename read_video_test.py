import streamlit as st
import subprocess
import cv2
import tensorflow as tf
from deepface import DeepFace
from retinaface import RetinaFace

video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

def retina(image):
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
        #emotion = DeepFace.analyze(face_image, actions=['emotion'],enforce_detection=False)

        # Display the emotion
        return face_image

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
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height),isColor = False)

    while True:
        ret,frame = cap.read()
        if not ret:
            break

        emotion_dominant = retina(frame)
        # Draw the emotion label on the frame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##<< Generates a grayscale (thus only one 2d-array)

        # Write the frame to the output video
        out_mp4.write(frame)

        # Show the processed frame (you can comment this out for faster processing)
        #cv2.imshow('Emotion Detection', frame)

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


