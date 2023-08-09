import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from deepface import DeepFace
from gradio_client import Client
from PIL import Image
import plotly.graph_objects as go
from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
from pysentimiento import create_analyzer
from annotated_text import annotated_text
from src.audio.melspec import plot_colored_polar
import mediapipe as mp


@dataclass
class EmotionRecognitionApp:
    """
    Emotion Recognition Code APP, accepts an audio file
    """

    def __init__(self):
        # Initialize necessary components
        self.analyzer = create_analyzer(task="emotion", lang="es")
        self.model = load_model("model3.h5")
        self.client = Client("https://088748f4f3a1603e6b.gradio.live")
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

    def log_file(self, txt=None):
        with open("log.txt", "a") as f:
            datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"{txt} - {datetoday};\n")

    def analyze_emotion(self, text):
        emotion_result = self.analyzer.predict(text)
        emotion = emotion_result.output
        probabilities = emotion_result.probas
        return emotion, probabilities

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

    def write_bytesio_to_file(self, filename, bytesio):
        with open(filename, "wb") as outfile:
            outfile.write(bytesio.getbuffer())

    def extract_audio(self, video_path, output_audio_path):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path)

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

    def main(self):
        side_img = Image.open("images/3603909.png")
        with st.sidebar:
            st.image(side_img, width=300)
            st.sidebar.subheader("Menú")
            website_menu = st.sidebar.selectbox("Menú",
                                                ("Reconocimiento de Emociones", "Descripción del Proyecto"))
            st.set_option('deprecation.showfileUploaderEncoding', False)

        if website_menu == "Reconocimiento de Emociones":
            st.sidebar.subheader("Modelo")
            model_type = st.sidebar.selectbox("¿Cómo le gustaría hacer la predicción?", ("mfccs", "mel-espectrogramas"))
            em3 = em6 = em7 = gender = False
            st.sidebar.subheader("Configuraciones")
            st.markdown("## Cargar el archivo")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                video_data = st.file_uploader("Upload file", ['mp4', 'mov', 'avi'])
                audio_file = None
                if video_data:
                    self.process_video(video_data)
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=self.timestamps, y=self.emotions_list, mode='markers+lines', name='Emotions',
                                   line=dict(color='blue', width=2)))
                    fig.update_layout(
                        title="Emotion Timeline",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Emotion",
                        template="plotly_white"
                    )
                    emotion_counts = dict(Counter(self.emotions_list))
                    fig_pie = go.Figure(
                        data=[go.Pie(labels=list(emotion_counts.keys()), values=list(emotion_counts.values()))])
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
                    audio_filename = './output_audio.wav'
                    audio_file = open(audio_filename,'rb')
                    if audio_file is not None:
                        if not os.path.exists("audio"):
                            os.makedirs("audio")
                        path = os.path.join("audio",audio_filename)
                            # extraer características
                            # mostrar el audio
                        st.audio(audio_file, format='audio/wav', start_time=0)
                        try:
                            wav, sr = librosa.load(path, sr=44100)
                            Xdb = self.get_melspec(path)[1]
                            mfccs = librosa.feature.mfcc(wav, sr=sr)
                            # # mostrar el audio
                            # st.audio(audio_file, format='audio/wav', start_time=0)
                        except Exception as e:
                            audio_file = None
                            st.error(f"Error {e} - formato incorrecto del archivo. Intente con otro archivo .wav.")
                    else:
                        st.error("Error desconocido")

                else:
                    if st.button("Probar con archivo de prueba"):
                        wav, sr = librosa.load("test.wav", sr=44100)
                        Xdb = self.get_melspec("test.wav")[1]
                        mfccs = librosa.feature.mfcc(wav, sr=sr)
                        # mostrar el audio
                        st.audio("test.wav", format='audio/wav', start_time=0)
                        path = "test.wav"
                        audio_file = "test"
            with col2:
                if audio_file is not None:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor('#d1d1e0')
                    plt.title("Forma de Onda")
                    librosa.display.waveplot(wav, sr=44100)
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    plt.gca().axes.spines["bottom"].set_visible(False)
                    plt.gca().axes.set_facecolor('#d1d1e0')
                    st.write(fig)
                else:
                    pass

        if model_type == "mfccs":
            em3 = st.sidebar.checkbox("3 emociones", True)
            em6 = st.sidebar.checkbox("6 emociones", True)
            em7 = st.sidebar.checkbox("7 emociones", True)
            whisper = st.sidebar.checkbox("Trancripción Whisper", True)
            gender = st.sidebar.checkbox("género", True)

        elif model_type == "mel-espectrogramas":
            st.sidebar.warning("Este modelo está temporalmente deshabilitado")

        else:
            st.sidebar.warning("Este modelo está temporalmente deshabilitado")

        if audio_file is not None:
            st.markdown("## Analizando...")
            if not audio_file == "test":
                st.sidebar.subheader("Archivo de audio")
                detalles_archivo = {"Nombre de archivo": audio_file.name, "Tamaño de archivo": audio_file.size}
                st.sidebar.write(detalles_archivo)
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor('#d1d1e0')
                    plt.title("MFCCs")
                    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig)
                with col2:
                    fig2 = plt.figure(figsize=(10, 2))
                    fig2.set_facecolor('#d1d1e0')
                    plt.title("Mel-log-espectrograma")
                    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig2)

            if model_type == "mfccs":
                st.markdown("## Predicciones")

                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    mfccs = self.get_mfccs(path, self.model.input_shape[-1])
                    mfccs = mfccs.reshape(1, *mfccs.shape)
                    pred = self.model.predict(mfccs)[0]

                    with col1:
                        if em3:
                            pos = pred[3] + pred[5] * .5
                            neu = pred[2] + pred[5] * .5 + pred[4] * .5
                            neg = pred[0] + pred[1] + pred[4] * .5
                            data3 = np.array([pos, neu, neg])
                            txt = "MFCCs\n" + self.get_title(data3, self.CAT3)
                            fig = plt.figure(figsize=(5, 5))
                            COLORS = self.color_dict(self.COLOR_DICT)
                            plot_colored_polar(fig, predictions=data3, categories=self.CAT3,
                                               title=txt, colors=COLORS)
                            st.write(fig)
                    with col2:

                        if em6:
                            txt = "MFCCs\n" + self.get_title(pred, self.CAT6)
                            fig2 = plt.figure(figsize=(5, 5))
                            COLORS = self.color_dict(self.COLOR_DICT)
                            plot_colored_polar(fig2, predictions=pred, categories=self.CAT6,
                                               title=txt, colors=COLORS)
                            st.write(fig2)

                    with col3:
                        if em7:
                            model_ = load_model("model4.h5")
                            mfccs_ = self.get_mfccs(path, model_.input_shape[-2])
                            mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
                            pred_ = model_.predict(mfccs_)[0]
                            txt = "MFCCs\n" + self.get_title(pred_, self.CAT7)
                            fig3 = plt.figure(figsize=(5, 5))
                            COLORS = self.color_dict(self.COLOR_DICT)
                            plot_colored_polar(fig3, predictions=pred_, categories=self.CAT7,
                                               title=txt, colors=COLORS)
                            st.write(fig3)

                    with col4:
                        if gender:
                            with st.spinner('Espera un momento...'):
                                gmodel = load_model("model_mw.h5")
                                gmfccs = self.get_mfccs(path, gmodel.input_shape[-1])
                                gmfccs = gmfccs.reshape(1, *gmfccs.shape)
                                gpred = gmodel.predict(gmfccs)[0]
                                gdict = [["mujer", "woman.png"], ["hombre", "man.png"]]
                                ind = gpred.argmax()
                                txt = "Género predicho: " + gdict[ind][0]
                                img = Image.open("images/" + gdict[ind][1])

                                fig4 = plt.figure(figsize=(3, 3))
                                fig4.set_facecolor('#d1d1e0')
                                plt.title(txt)
                                plt.imshow(img)
                                plt.axis("off")
                                st.write(fig4)
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        if whisper:
                            with st.spinner('Procesando Transcripción'):
                                result = self.client.predict(
                                    "medium",
                                    "Spanish",
                                    "",
                                    [f"./audio/{audio_file.name}"],
                                    "",
                                    "transcribe",
                                    "none",
                                    5,
                                    5,
                                    False,
                                    False,
                                    api_name="/predict"
                                )
                                # Split the data_string into filepaths and text_data
                                text_data = result[2]

                                # Skip the WEBVTT header and start processing from the first timestamp
                                lines = text_data.split("\n")
                                start_index = 0
                                while start_index < len(lines):
                                    if "-->" in lines[start_index]:
                                        break
                                    start_index += 1

                                # Show the word cloud
                                st.subheader("Word Cloud")
                                self.create_word_cloud("\n".join(lines[start_index + 1:]))

                                st.subheader("Emoción obtenida apartir de la transcripción de texto:")
                                for i in range(start_index, len(lines), 3):
                                    if i + 2 < len(
                                            lines):  # Check if there are enough lines to extract timestamp and text
                                        timestamp = lines[i].strip()
                                        text = lines[i + 1].strip()
                                        st.markdown(f"{timestamp}")
                                        st.divider()
                                        if text is not " ":
                                            emotion_result, probabilities = self.analyze_emotion(text)
                                            annotated_text((text, emotion_result))
                                            st.write(f"Análisis emocional: {emotion_result}\n")
                                            st.plotly_chart(self.plot_emotion_probabilities(probabilities))


if __name__ == '__main__':
    app = EmotionRecognitionApp()
    app.main()
