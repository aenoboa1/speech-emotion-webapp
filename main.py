from read_face_main import EmotionDetectionApp

from datetime import datetime
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gradio_client import Client
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
from pysentimiento import create_analyzer
from annotated_text import annotated_text
from src.audio.melspec import plot_colored_polar
from collections import Counter
import subprocess
import numpy as np
import plotly.io as pio
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from deepface import DeepFace

def main():
    st.title("Emotion Recognition & Detection App")
    app_mode = st.selectbox(
        "Select an app",
        ["Emotion Recognition", "Emotion Detection"]
    )

    if app_mode == "Emotion Recognition":
        recognition_app = EmotionRecognitionApp()
        recognition_app.run()

    if app_mode == "Emotion Detection":
        detection_app = EmotionDetectionApp()
        detection_app.run()