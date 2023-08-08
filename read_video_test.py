from collections import Counter
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import subprocess
import cv2
from deepface import DeepFace
import mediapipe as mp

class EmotionDetectionApp:
    def __init__(self):
        self.temp_file_to_save = './temp_file_1.mp4'
        self.temp_file_result = './temp_file_2.mp4'
        self.cap = None
        self.out_mp4 = None
        self.emotions_list = []
        self.timestamps = []

    def write_bytesio_to_file(self, filename, bytesio):
        with open(filename, "wb") as outfile:
            outfile.write(bytesio.getbuffer())

    def detect_emotion(self, face_image):
        if face_image is None or face_image.size == 0:
            return None
        emotions_list = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        if not emotions_list:
            return None
        emotions = emotions_list[0]
        emotion_label = emotions['dominant_emotion']
        return emotion_label

    def mediapipe_face_detection(self, image):
        mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        results = mp_face_detection.process(image)

        if results.detections:
            ih, iw, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                w = min(iw - x, w)
                h = min(ih - y, h)
                if w > 0 and h > 0:
                    face_image = image[y:y + h, x:x + w]
                    if not isinstance(face_image, np.ndarray) or face_image.shape[-1] != 3:
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    emotion_label = self.detect_emotion(face_image)
                    current_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if emotion_label is not None:
                        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

    def process_video(self, video_data):
        self.write_bytesio_to_file(self.temp_file_to_save, video_data)
        self.cap = cv2.VideoCapture(self.temp_file_to_save)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_mp4 = cv2.VideoWriter(self.temp_file_result, fourcc_mp4, frame_fps, (width, height), isColor=False)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            emotion_dominant = self.mediapipe_face_detection(frame)
            emotion_label = self.detect_emotion(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if emotion_label is not None:
                self.emotions_list.append(emotion_label)
                self.timestamps.append(current_timestamp)
            self.out_mp4.write(gray)

        self.cap.release()
        self.out_mp4.release()

    def convert_to_h264(self):
        converted_video = "./testh264.mp4"
        subprocess.call(args=f"ffmpeg -y -i {self.temp_file_result} -c:v libx264 {converted_video}".split(" "))

    def run(self):
        st.title("Emotion Detection App")
        video_data = st.file_uploader("Upload file", ['mp4', 'mov', 'avi'])
        if video_data:
            self.process_video(video_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.timestamps, y=self.emotions_list, mode='markers+lines', name='Emotions', line=dict(color='blue', width=2)))
            fig.update_layout(
                title="Emotion Timeline",
                xaxis_title="Time (seconds)",
                yaxis_title="Emotion",
                template="plotly_white"
            )

            emotion_counts = dict(Counter(self.emotions_list))
            fig_pie = go.Figure(data=[go.Pie(labels=list(emotion_counts.keys()), values=list(emotion_counts.values()))])
            fig_pie.update_layout(
                title="Emotion Summary",
                template="plotly_white"
            )

            st.plotly_chart(fig)
            st.plotly_chart(fig_pie)

            self.convert_to_h264()

            col1, col2 = st.columns(2)
            col1.header("Original Video")
            col1.video(self.temp_file_to_save)
            col2.header("Output from OpenCV (MPEG-4)")
            col2.video(self.temp_file_result)
            col2.header("After conversion to H264")
            col2.video("./testh264.mp4")

if __name__ == "__main__":
    app = EmotionDetectionApp()
    app.run()
