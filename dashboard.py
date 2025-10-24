# =====================================================
# 👟 Gender & Footwear AI Detection Dashboard 
# =====================================================
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="👟 Gender & Footwear AI Detection",
    layout="wide",
    page_icon="👟"
)

st.title("👟 Gender & Footwear Recognition App")
st.markdown("""
Temukan kecocokan gaya kamu! 👟 Aplikasi ini bisa mengenali gender dan jenis alas kaki secara otomatis hanya dari satu gambar  
""")

# =====================================================
# 🎨 SIDEBAR - INFORMASI APLIKASI
# =====================================================
with st.sidebar.expander("ℹ️ Tentang Aplikasi", expanded=True):
    st.markdown("""
    ## 👟 Gender & Footwear Recognition App  
    Aplikasi ini dibuat sebagai **proyek analisis berbasis kecerdasan buatan (AI)**  
    untuk mengenali **jenis kelamin (gender)** dan **jenis alas kaki (footwear)** dari gambar atau tangkapan kamera secara otomatis.

    ---
    ### 🎯 Tujuan Aplikasi
    - Menerapkan konsep **Computer Vision** secara interaktif dan praktis  
    - Memberikan pengalaman **deteksi visual real-time** bagi pengguna  
    - Menunjukkan bagaimana **AI dapat digunakan dalam klasifikasi objek sederhana**

    ---
    ### 🔍 Fitur Utama:
    - 🧍 **Deteksi Gender** → Men / Women  
    - 👞 **Klasifikasi Alas Kaki** → Boot / Sandal / Sepatu  
    - 💾 **Simpan Riwayat ke CSV** → untuk dokumentasi hasil deteksi  
    - 📊 **Tampilkan Statistik Visual** → grafik hasil prediksi  

    ---
    ### 🧠 Tentang Teknologi
    Aplikasi ini menggunakan kombinasi **deteksi objek** dan **klasifikasi citra**  
    untuk menghasilkan hasil deteksi yang cepat dan akurat.
    """)

# =====================================================
# 🌈 PILIHAN TEMA WARNA
# =====================================================
st.sidebar.markdown("### 🌈 Tema Tampilan")
theme = st.sidebar.selectbox(
    "Pilih Tema",
    ["💡 Default", "🌙 Dark", "🌊 Ocean", "🌸 Pink", "🌲 Forest", "🌅 Sunset"]
)

# Tema CSS custom
theme_styles = {
    "🌙 Dark": {
        "body": "#1E1E1E", "text": "#EAEAEA", "sidebar": "#2B2B2B", "button": "#444", "button_hover": "#555", "link": "#9CDCFE"
    },
    "🌊 Ocean": {
        "body": "#E0F7FA", "text": "#004D40", "sidebar": "#B2EBF2", "button": "#80DEEA", "button_hover": "#4DD0E1", "link": "#00838F"
    },
    "🌸 Pink": {
        "body": "#FFF0F6", "text": "#4A0033", "sidebar": "#FFD6E7", "button": "#FFB6C1", "button_hover": "#FF9EBB", "link": "#C2185B"
    },
    "🌲 Forest": {
        "body": "#E8F5E9", "text": "#1B5E20", "sidebar": "#C8E6C9", "button": "#A5D6A7", "button_hover": "#81C784", "link": "#2E7D32"
    },
    "🌅 Sunset": {
        "body": "#FFF3E0", "text": "#4E342E", "sidebar": "#FFE0B2", "button": "#FFCC80", "button_hover": "#FFB74D", "link": "#E65100"
    },
    "💡 Default": {
        "body": "#FAFAFA", "text": "#0E1117", "sidebar": "#FFFFFF", "button": "#AEDFF7", "button_hover": "#90CAF9", "link": "#1E88E5"
    }
}

style = theme_styles[theme]
st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {style['body']} !important; color: {style['text']} !important; }}
    section[data-testid="stSidebar"] {{ background-color: {style['sidebar']} !important; color: {style['text']} !important; }}
    .stButton>button {{ background-color: {style['button']} !important; color: {style['text']} !important; border-radius: 8px; }}
    .stButton>button:hover {{ background-color: {style['button_hover']} !important; transform: scale(1.03); }}
    a {{ color: {style['link']} !important; }}
    a:hover {{ text-decoration: underline; }}
    </style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource(show_spinner=False)
def load_models():
    yolo_path = "model/Leni Gustia_Laporan 4.pt"
    cnn_path = "model/Leni_Gustia_Laporan_2.h5"

    if not os.path.exists(yolo_path):
        st.error(f"❌ File YOLO tidak ditemukan: {yolo_path}")
        st.stop()
    if not os.path.exists(cnn_path):
        st.error(f"❌ File CNN tidak ditemukan: {cnn_path}")
        st.stop()

    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(cnn_path)
    class_labels = ["Boot", "Sandal", "Shoe"]

    return yolo_model, classifier, class_labels

yolo_model, classifier, class_labels = load_models()

# =====================================================
# DETEKSI GENDER (YOLO)
# =====================================================
def detect_objects(img, conf_threshold=0.3):
    results = yolo_model(img)
    annotated_img = results[0].plot()
    detected_objects = []
    valid_labels = ["Men", "Women"]

    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = results[0].names[cls]
        if conf >= conf_threshold:
            detected_objects.append({
                "label": label if label in valid_labels else "Objek tidak sesuai domain gender",
                "confidence": round(conf * 100, 2)
            })
    return annotated_img, detected_objects

# =====================================================
# KLASIFIKASI ALAS KAKI (CNN)
# =====================================================
def classify_image(img):
    try:
        img = img.convert("RGB")
        input_shape = classifier.input_shape[1:3]
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        class_name = class_labels[class_index]

        if confidence < 0.6:
            return "Bukan alas kaki", round(confidence * 100, 2)
        return class_name, round(confidence * 100, 2)
    except Exception as e:
        st.error(f"⚠ Error klasifikasi: {e}")
        return "Bukan alas kaki", 0

# =====================================================
# SIDEBAR SETTINGS
# =====================================================
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold (YOLO)", 0.1, 1.0, 0.3, 0.05)
export_enable = st.sidebar.checkbox("💾 Simpan Riwayat ke CSV", value=True)
show_chart = st.sidebar.checkbox("📈 Tampilkan Statistik Visual", value=True)
input_mode = st.sidebar.radio("Metode Input:", ["📤 Upload Gambar", "📷 Gunakan Kamera"])

uploaded_file = None
if input_mode == "📤 Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
elif input_mode == "📷 Gunakan Kamera":
    camera_input = st.camera_input("📸 Ambil gambar langsung dari kamera")
    if camera_input:
        uploaded_file = camera_input
        st.caption("💡 Pastikan izin kamera sudah diberikan di browser.")

# =====================================================
# INISIALISASI DATA SESI
# =====================================================
if "detections" not in st.session_state:
    st.session_state.detections = {"gender": 0, "footwear": 0}
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# MODE DETEKSI & KLASIFIKASI
# =====================================================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "🧍 Deteksi Gender (YOLO)":
        st.subheader("🧍 Deteksi Gender (Men/Women)")
        with st.spinner("🔎 Mendeteksi gender..."):
            start_time = time.time()
            annotated_img, detections = detect_objects(img, conf_threshold)
            duration = time.time() - start_time

        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
        st.caption(f"⏱ Waktu Proses: {duration:.2f} detik")

        if detections:
            valid = any(d["label"] in ["Men", "Women"] for d in detections)
            if valid:
                st.session_state.detections["gender"] += 1
                gender_detected = detections[0]["label"]
                st.session_state.history.append({"Tipe": "Gender", "Hasil": gender_detected})

                # Rekomendasi
                if gender_detected == "Men":
                    st.markdown("### 🧴 Rekomendasi untuk Pria")
                    st.info("- Gunakan *moisturizer* harian.\n- Outfit kasual: kemeja + jeans.\n- Face Wash Men Deep Clean - Rp 35.000")
                elif gender_detected == "Women":
                    st.markdown("### 💅 Rekomendasi untuk Wanita")
                    st.info("- Gunakan *sunscreen* setiap hari.\n- Outfit kasual: floral dress.\n- Serum Vitamin C Bright - Rp 50.000")
            else:
                st.warning("⚠ Gambar bukan domain gender.")
        else:
            st.warning("⚠ Tidak ada objek terdeteksi.")

    elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
        st.subheader("👞 Klasifikasi Alas Kaki (Boot/Sandal/Shoe)")
        annotated_img, detections = detect_objects(img, conf_threshold)
        contains_human = any(d["label"] in ["Men", "Women"] for d in detections)

        if contains_human:
            st.error("🚫 Gambar mengandung manusia, bukan domain alas kaki.")
        else:
            with st.spinner("🧠 Mengklasifikasikan..."):
                start_time = time.time()
                class_name, confidence = classify_image(img)
                duration = time.time() - start_time

            st.caption(f"⏱ Waktu Proses: {duration:.2f} detik")

            if class_name == "Bukan alas kaki":
                st.warning("⚠ Gambar tidak dikenali sebagai alas kaki.")
            else:
                st.session_state.detections["footwear"] += 1
                st.session_state.history.append({"Tipe": "Alas Kaki", "Hasil": class_name})
                st.success(f"✅ Jenis Alas Kaki: *{class_name}* ({confidence}%)")
else:
    st.info("📤 Silakan unggah gambar atau gunakan kamera.")

# =====================================================
# STATISTIK DETEKSI
# =====================================================
st.markdown("---")
st.markdown("### 📊 Statistik Deteksi Sementara")

col1, col2 = st.columns(2)
col1.metric("Total Deteksi Gender", st.session_state.detections["gender"])
col2.metric("Total Deteksi Alas Kaki", st.session_state.detections["footwear"])

if show_chart and len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    counts = df["Hasil"].value_counts()

    theme_colors = {
        "🌸 Pink": ["#FF80AB", "#F48FB1", "#EC407A"],
        "🌲 Forest": ["#66BB6A", "#81C784", "#A5D6A7"],
        "🌊 Ocean": ["#4DD0E1", "#26C6DA", "#00ACC1"],
        "🌅 Sunset": ["#FFB74D", "#FF8A65", "#F06292"],
        "🌙 Dark": ["#90CAF9", "#F48FB1", "#CE93D8"],
        "💡 Default": ["#64B5F6", "#4FC3F7", "#81D4FA"]
    }
    colors = theme_colors.get(theme, ["#64B5F6", "#4FC3F7", "#81D4FA"])

    fig, ax = plt.subplots()
    counts.plot(kind="bar", color=colors, ax=ax)
    ax.set_xlabel("Kategori Deteksi", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)
    ax.set_title("Statistik Deteksi (Gender & Alas Kaki)", fontsize=14, weight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

# =====================================================
# EKSPOR DATA
# =====================================================
if export_enable and len(st.session_state.history) > 0:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Unduh Riwayat Deteksi (CSV)", data=csv, file_name="riwayat_deteksi.csv", mime="text/csv")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 14px;'>
    © 2025 <b>Smart AI Vision</b> — Created by <b>Leni Gustia 👩‍💻</b><br>
    <a href='https://www.linkedin.com/in/leni-gustia-405a40294/' target='_blank' style='text-decoration: none; color: #008B8B; font-weight: bold;'>
        🔗 LinkedIn: leni-gustia-405a40294
    </a><br>
    Built with ❤️ using YOLOv8 + TensorFlow CNN + Streamlit
</div>
""", unsafe_allow_html=True)
