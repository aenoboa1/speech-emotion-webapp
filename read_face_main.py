import os
from collections import Counter
from datetime import datetime

import librosa
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio  # Import plotly.io for locale setting
import subprocess
import cv2
from deepface import DeepFace
import mediapipe as mp
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from wordcloud import WordCloud


class EmotionDetectionApp:
    """
    Main class for EmotionDetection
    """

    def __init__(self):
        self.model = load_model("model3.h5")
        self.client = Client("https://b33d5ab7b72a667f58.gradio.live/")
        self.starttime = datetime.now()
        self.CAT6 = ['miedo', 'enojo', 'neutral', 'feliz', 'triste', 'sorpresa']
        self.CAT7 = ['miedo', 'asco', 'neutral', 'feliz', 'triste', 'sorpresa', 'enojo']
        self.CAT3 = ["positivo", "neutral", "negativo"]
        self.COLOR_DICT = {
            "neutral": "grey",
            "positivo": "green",
            "feliz": "green",
            "sorpresa": "orange",
            "miedo": "purple",
            "negativo": "red",
            "enojo": "red",
            "triste": "lightblue",
            "asco": "brown"
        }
        self.TEST_CAT = ['miedo', 'asco', 'neutral', 'feliz', 'triste', 'sorpresa', 'enojo']
        self.TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])
        self.hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
        self.hide_decoration_bar_style = '''
            <style>
                header {visibility: hidden;}
            </style>
        '''
        self.temp_file_to_save = './temp_file_1.mp4'
        self.temp_file_result = './temp_file_2.mp4'
        self.cap = None
        self.out_mp4 = None
        self.emotions_list = []
        self.timestamps = []


    def write_bytesio_to_file(self, filename, bytesio):
        with open(filename, "wb") as outfile:
            outfile.write(bytesio.getbuffer())

    def extract_audio(self, video_path, output_audio_path):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path)

    def log_file(self, txt=None):
        with open("log.txt", "a") as f:
            datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"{txt} - {datetoday};\n")

    # def analyze_emotion(self, text):
    #    emotion_result = self.analyzer.predict(text)
    #    emotion = emotion_result.output
    #    probabilities = emotion_result.probas
    #    return emotion, probabilities

    def plot_emotion_probabilities(self, probabilities):
        emotions_emoji_dict = {
            "anger": ("😠", "red"),
            "disgust": ("🤮", "purple"),
            "fear": ("😨😱", "orange"),
            "joy": ("😂", "yellow"),
            "others": ("😐", "gray"),
            "sadness": ("😔", "blue"),
            "surprise": ("😮", "cyan")
        }
        emotions = list(probabilities.keys())
        probabilities = list(probabilities.values())
        fig = go.Figure(go.Bar(x=emotions, y=probabilities,
                               marker=dict(color=[emotions_emoji_dict[emotion][1] for emotion in emotions])))
        fig.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title="Probability"),
            title="Emotion Probabilities",
            showlegend=False
        )
        return fig

    def save_audio(self, file):
        if file.size > 40000000:
            return 1

        folder = "audio"
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('No se pudo eliminar %s. Razón: %s' % (file_path, e))

        try:
            with open("log0.txt", "a") as f:
                f.write(f"{file.name} - {file.size} - {datetoday};\n")
        except:
            pass

        with open(os.path.join(folder, file.name), "wb") as f:
            f.write(file.getbuffer())
        return 0

    def get_melspec(self, audio):
        y, sr = librosa.load(audio, sr=44100)
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        img = np.stack((Xdb,) * 3, -1)
        img = img.astype(np.uint8)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.resize(grayImage, (224, 224))
        rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
        return rgbImage, Xdb

    @staticmethod
    def get_mfccs(self, audio, limit):
        y, sr = librosa.load(audio)
        a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
        if a.shape[1] > limit:
            mfccs = a[:, :limit]
        elif a.shape[1] < limit:
            mfccs = np.zeros((a.shape[0], limit))
            mfccs[:, :a.shape[1]] = a
        return mfccs

    @st.cache_resource
    def get_title(self, predictions, categories):
        title = f"Emoción Detectada: {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
        return title

    @st.cache_resource
    def color_dict(self, coldict):
        return coldict

    @st.cache_resource
    def plot_polar(self, fig, predictions, categories, title, colors):
        N = len(predictions)
        ind = predictions.argmax()
        color_sector = colors[categories[ind]]
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = np.zeros_like(predictions)
        radii[predictions.argmax()] = predictions.max() * 10
        width = np.pi / 1.8 * predictions
        fig.set_facecolor("#d1d1e0")
        ax = plt.subplot(111, polar="True")
        ax.bar(theta, radii, width=width, bottom=0.0, color=color_sector, alpha=0.25)
        angles = [i / float(N) * 2 * np.pi for i in range(N)]
        angles += angles[:1]
        data = list(predictions)
        data += data[:1]
        plt.polar(angles, data, color=color_sector, linewidth=2)
        plt.fill(angles, data, facecolor=color_sector, alpha=0.25)
        ax.spines['polar'].set_color('lightgrey')
        ax.set_theta_offset(np.pi / 3)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
        plt.suptitle(title, color="darkblue", size=12)
        plt.title(f"BIG {N}\n", color=color_sector)
        plt.ylim(0, 1)
        plt.subplots_adjust(top=0.75)

    def create_word_cloud(self, text_data: object):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        st.image(wordcloud.to_array(), use_column_width=True)
    def detect_emotion(self, face_image):
        if face_image is None or face_image.size == 0:
            return None
        emotions_list = DeepFace.analyze(
            face_image,
            actions=['emotion'],
            detector_backend="skip",
            enforce_detection=False)
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
            fig.add_trace(go.Scatter(x=self.timestamps, y=self.emotions_list, mode='markers+lines', name='Emotions',
                                     line=dict(color='blue', width=2)))
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
            col1.header("Video original")
            col1.video(self.temp_file_to_save)
            col2.header("Video con emociones procesadas")
            col2.video("./testh264.mp4")
            self.extract_audio("./temp_file_1.mp4", "output_audio.wav")


if __name__ == "__main__":
    app = EmotionDetectionApp()
    app.run()
