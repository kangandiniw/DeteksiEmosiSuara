import numpy as np
import streamlit as st
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
from melspec import plot_colored_polar, plot_melspec

# load models
model = load_model("model3.h5")

# constants
starttime = datetime.now()

CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {"neutral": "grey",
              "positive": "green",
              "happy": "green",
              "surprise": "orange",
              "fear": "purple",
              "negative": "red",
              "angry": "red",
              "sad": "lightblue",
              "disgust": "brown"}

TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])

# page settings
st.set_page_config(page_title="Deteksi Emosi Suara", page_icon=":speech_balloon:", layout="wide")

# @st.cache
def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")

# @st.cache
def save_audio(file):
    if file.size > 4000000:
        return 1
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass
    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0

# @st.cache
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

@st.cache
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
    return title

@st.cache
def color_dict(coldict=COLOR_DICT):
    return COLOR_DICT

@st.cache
def plot_polar(fig, predictions=TEST_PRED, categories=TEST_CAT,
               title="TEST", colors=COLOR_DICT):
    N = len(predictions)
    ind = predictions.argmax()
    COLOR = color_sector = colors[categories[ind]]
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
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)
    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.suptitle(title, color="darkblue", size=12)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)

def main():

    
    st.markdown("<h1 style='text-align: center;'>Aplikasi Web Deteksi Emosi Suara</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Analisis dan Deteksi Emosi pada Audio</h3>", unsafe_allow_html=True)
        

    side_img = Image.open("images/emsoi.jpeg")
    with st.sidebar:
        st.image(side_img, width=300)
    st.sidebar.subheader("Menu")
    website_menu = st.sidebar.selectbox("Menu", ("Deteksi Emosi Suara", "Anggota Kelompok"))
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if website_menu == "Deteksi Emosi Suara":
        st.sidebar.subheader("Model Settings")
        model_type = "mfccs"
        em3 = st.sidebar.checkbox("3 Emosi", True)
        em6 = st.sidebar.checkbox("6 Emosi")
        em7 = st.sidebar.checkbox("7 Emosi")
        st.markdown("## Upload the file")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    if_save_audio = save_audio(audio_file)
                    if if_save_audio == 1:
                        st.warning("Ukuran file terlalu besar. Coba unggah file lain.")
                    elif if_save_audio == 0:
                        st.audio(audio_file, format='audio/wav', start_time=0)
                        try:
                            wav, sr = librosa.load(path, sr=44100)
                            mfccs = librosa.feature.mfcc(wav, sr=sr)
                            st.success("Berkas berhasil diunggah!")
                        except Exception as e:
                            audio_file = None
                            st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                    else:
                        st.error("Unknown error")
                else:
                    if st.button("Percobaan (Demo)"):
                        wav, sr = librosa.load("aslam_marah.wav", sr=44100)
                        mfccs = librosa.feature.mfcc(wav, sr=sr)
                        st.audio("aslam_marah.wav", format='audio/wav', start_time=0)
                        path = "aslam_marah.wav"
                        audio_file = "test"

        if audio_file is not None:
            st.markdown("## Menganalisis...")
            if not audio_file == "test":
                st.sidebar.subheader("Audio file")
                file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
                st.sidebar.write(file_details)

            with st.container():
                col1, _ = st.columns([1, 1])

                with col1:
                    fig_waveform, ax_waveform = plt.subplots(figsize=(10, 2))
                    ax_waveform.set_facecolor('#d1d1e0')
                    ax_waveform.set_title("Wave-form")
                    librosa.display.waveplot(wav, sr=44100, ax=ax_waveform)
                    ax_waveform.get_yaxis().set_visible(False)
                    ax_waveform.get_xaxis().set_visible(False)
                    ax_waveform.spines["right"].set_visible(False)
                    ax_waveform.spines["left"].set_visible(False)
                    ax_waveform.spines["top"].set_visible(False)
                    ax_waveform.spines["bottom"].set_visible(False)
                    ax_waveform.set_facecolor('#d1d1e0')
                    plt.tight_layout()
                    st.pyplot(fig_waveform)

                    fig_mfccs, ax_mfccs = plt.subplots(figsize=(10, 4))
                    ax_mfccs.set_facecolor('#d1d1e0')
                    ax_mfccs.set_title("MFCCs")
                    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax_mfccs)
                    ax_mfccs.get_yaxis().set_visible(False)
                    ax_mfccs.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_mfccs)
                    
            if model_type == "mfccs":
                st.markdown("## Hasil Prediksi")
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    mfccs = get_mfccs(path, model.input_shape[-1])
                    mfccs = mfccs.reshape(1, *mfccs.shape)
                    pred = model.predict(mfccs)[0]

                    with col1:
                        if em3:
                            pos = pred[3] + pred[5] * .5
                            neu = pred[2] + pred[5] * .5 + pred[4] * .5
                            neg = pred[0] + pred[1] + pred[4] * .5
                            data3 = np.array([pos, neu, neg])
                            txt = "MFCCs\n" + get_title(data3, CAT3)
                            fig = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig, predictions=data3, categories=CAT3,
                                               title=txt, colors=COLORS)
                            st.write(fig)
                    with col2:
                        if em6:
                            txt = "MFCCs\n" + get_title(pred, CAT6)
                            fig2 = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig2, predictions=pred, categories=CAT6,
                                               title=txt, colors=COLORS)
                            st.write(fig2)
                    with col3:
                        if em7:
                            model_ = load_model("model4.h5")
                            mfccs_ = get_mfccs(path, model_.input_shape[-2])
                            mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
                            pred_ = model_.predict(mfccs_)[0]
                            txt = "MFCCs\n" + get_title(pred_, CAT7)
                            fig3 = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig3, predictions=pred_, categories=CAT7,
                                               title=txt, colors=COLORS)
                            st.write(fig3)

    elif website_menu == "Anggota Kelompok":
        st.subheader("Anggota Kelompok")

        # Kelompok Kita
        col1, col2 = st.columns([3, 2])

        with col1:
            st.info("Ferris Tita Sabilillah - A11.2021.13579")
            st.info("Aslam Thariq Akbar Akrami - A11.2021.13224")
            st.info("Immanuel Felix Abel Ketaren - A11.2021.13676")
            st.info("Bastiaans, Jessica Carmelita - A11.2021.13249")
            st.info("Kang, Andini Wulandari - A11.2021.13273")


if __name__ == '__main__':
    main()