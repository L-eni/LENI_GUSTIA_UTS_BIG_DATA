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
from PIL import ImageEnhance, ImageOps, ImageFilter

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
    Aplikasi ini bertujuan untuk:
    - Menerapkan konsep **Computer Vision** secara interaktif dan praktis.  
    - Memberikan pengalaman **deteksi visual real-time** bagi pengguna.  
    - Menunjukkan bagaimana **AI dapat digunakan dalam klasifikasi objek sederhana.**

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

    ---
    Pilih mode, unggah gambar, dan lihat hasil deteksinya secara real-time! 🚀
    """)

# =====================================================
# 🌈 PILIHAN TEMA WARNA (PASTEL STYLE)
# =====================================================
st.sidebar.markdown("### 🌈 Tema Tampilan")
theme = st.sidebar.selectbox(
    "Pilih Tema",
    ["💡 Default", "🌙 Dark", "🌊 Ocean", "🌸 Pink", "🌲 Forest", "🌅 Sunset"]
)

# --- 🌙 DARK (soft gray-black) ---
if theme == "🌙 Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #1E1E1E !important; color: #EAEAEA !important; }
        section[data-testid="stSidebar"] { background-color: #2B2B2B !important; color: #EAEAEA !important; }
        .stButton>button { background-color: #444 !important; color: #FAFAFA !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #555 !important; transform: scale(1.03); }
        a { color: #9CDCFE !important; }
        a:hover { color: #C5E4FD !important; text-decoration: underline; }
        </style>
    """, unsafe_allow_html=True)

# --- 🌊 OCEAN (pastel toska & biru muda) ---
elif theme == "🌊 Ocean":
    st.markdown("""
        <style>
        body, .stApp { background-color: #E0F7FA !important; color: #004D40 !important; }
        section[data-testid="stSidebar"] { background-color: #B2EBF2 !important; color: #004D40 !important; }
        .stButton>button { background-color: #80DEEA !important; color: #004D40 !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #4DD0E1 !important; }
        a { color: #00838F !important; }
        a:hover { color: #006064 !important; text-decoration: underline; }
        </style>
    """, unsafe_allow_html=True)

# --- 🌸 PINK (soft blush pastel) ---
elif theme == "🌸 Pink":
    st.markdown("""
        <style>
        body, .stApp { background-color: #FFF0F6 !important; color: #4A0033 !important; }
        section[data-testid="stSidebar"] { background-color: #FFD6E7 !important; color: #4A0033 !important; }
        .stButton>button { background-color: #FFB6C1 !important; color: #4A0033 !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #FF9EBB !important; }
        a { color: #C2185B !important; }
        a:hover { color: #AD1457 !important; text-decoration: underline; }
        </style>
    """, unsafe_allow_html=True)

# --- 🌲 FOREST (soft mint & hijau muda) ---
elif theme == "🌲 Forest":
    st.markdown("""
        <style>
        body, .stApp { background-color: #E8F5E9 !important; color: #1B5E20 !important; }
        section[data-testid="stSidebar"] { background-color: #C8E6C9 !important; color: #1B5E20 !important; }
        .stButton>button { background-color: #A5D6A7 !important; color: #1B5E20 !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #81C784 !important; }
        a { color: #2E7D32 !important; }
        a:hover { color: #1B5E20 !important; text-decoration: underline; }
        </style>
    """, unsafe_allow_html=True)

# --- 🌅 SUNSET (peach pastel & coral lembut) ---
elif theme == "🌅 Sunset":
    st.markdown("""
        <style>
        body, .stApp { background-color: #FFF3E0 !important; color: #4E342E !important; }
        section[data-testid="stSidebar"] { background-color: #FFE0B2 !important; color: #4E342E !important; }
        .stButton>button { background-color: #FFCC80 !important; color: #4E342E !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #FFB74D !important; }
        a { color: #E65100 !important; }
        a:hover { color: #BF360C !important; text-decoration: underline; }
        </style>
    """, unsafe_allow_html=True)

# --- 💡 DEFAULT (putih lembut & abu pastel) ---
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #FAFAFA !important; color: #0E1117 !important; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF !important; color: #0E1117 !important; }
        .stButton>button { background-color: #AEDFF7 !important; color: #003366 !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #90CAF9 !important; }
        a { color: #1E88E5 !important; }
        a:hover { color: #1565C0 !important; text-decoration: underline; }
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
# ====================================================

def classify_image(img):
    try:
        img = img.convert("RGB")

        # 🧩 Preprocessing adaptif biar hasil stabil di semua kondisi
        img = ImageOps.autocontrast(img, cutoff=2)
        img = img.filter(ImageFilter.SMOOTH_MORE)

        # Penyesuaian brightness & kontras ringan
        enhancer_b = ImageEnhance.Brightness(img)
        img = enhancer_b.enhance(0.9)  # bisa dicoba 0.8–1.2

        enhancer_c = ImageEnhance.Contrast(img)
        img = enhancer_c.enhance(1.2)  # bisa dicoba 1.0–1.3

        # Resize ke ukuran input model
        input_shape = classifier.input_shape[1:3]
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        prediction = classifier.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        class_name = class_labels[class_index]

        # 🔄 Fallback: kalau confidence rendah, coba grayscale ulang
        if confidence < 0.55:
            gray = img.convert("L").convert("RGB")
            gray_resized = gray.resize(input_shape)
            gray_array = image.img_to_array(gray_resized)
            gray_array = np.expand_dims(gray_array, axis=0) / 255.0

            re_pred = classifier.predict(gray_array, verbose=0)
            re_conf = np.max(re_pred)
            re_class = class_labels[np.argmax(re_pred)]

            if re_conf > confidence:
                class_name, confidence = re_class, re_conf

        # Filter akhir
        if confidence < 0.45:
            return "Bukan alas kaki", round(confidence * 100, 2)

        return class_name, round(confidence * 100, 2)

    except Exception as e:
        st.error(f"⚠ Error klasifikasi: {e}")
        return "Bukan alas kaki", 0

# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold (YOLO)", 0.1, 1.0, 0.3, 0.05)

st.sidebar.markdown("### ⚙️ Fitur Opsional")
export_enable = st.sidebar.checkbox("💾 Simpan Riwayat ke CSV", value=True)
show_chart = st.sidebar.checkbox("📈 Tampilkan Statistik Visual", value=True)

st.sidebar.markdown("### Pilih Sumber Gambar")
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

# =========================
# Fungsi helper
# =========================
def contains_human(img, conf_threshold=0.5):
    """
    Mengembalikan True jika YOLO mendeteksi manusia (Men/Women) dalam gambar.
    """
    _, detections = detect_objects(img, conf_threshold)
    return any(d["label"] in ["Men", "Women"] for d in detections)

# =====================================================
# MODE: YOLO (Deteksi Gender)
# =====================================================
if menu == "🧍 Deteksi Gender (YOLO)":
    st.subheader("🧍 Deteksi Gender (Men/Women)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔎 Mendeteksi gender..."):
            start_time = time.time()
            annotated_img, detections = detect_objects(img, conf_threshold)
            duration = time.time() - start_time

        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
        st.caption(f"⏱ Waktu Proses: {duration:.2f} detik")

        human_detected = any(d["label"] in ["Men", "Women"] for d in detections)

        if human_detected:
            gender_detected = detections[0]["label"]
            st.session_state.detections["gender"] += 1
            st.session_state.history.append({"Tipe": "Gender", "Hasil": gender_detected})

            if gender_detected == "Men":
                st.markdown("### 🧴 Rekomendasi untuk Pria")
                st.info("""
                - Gunakan *moisturizer* harian untuk menjaga kelembapan kulit.  
                - Pilihan outfit kasual: **kemeja polos + jeans slim fit**.  
                - Produk rekomendasi utama: **Face Wash Men Deep Clean - Rp 35.000**
                """)
            elif gender_detected == "Women":
                st.markdown("### 💅 Rekomendasi untuk Wanita")
                st.info("""
                - Gunakan *sunscreen* setiap hari (minimal SPF 30+) untuk melindungi kulit dari UV.  
                - Coba gaya kasual dengan **floral dress** dan aksesori minimalis.  
                - Produk rekomendasi utama: **Serum Vitamin C Bright - Rp 50.000**
                """)
        else:
            st.warning("⚠ Tidak ada manusia terdeteksi dalam gambar.")
    else:
        st.info("📤 Silakan unggah gambar atau gunakan kamera.")

# =====================================================
# MODE: CNN (Klasifikasi Alas Kaki)
# =====================================================
# =====================================================
# MODE: CNN (Klasifikasi Alas Kaki)
# =====================================================
elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
    st.subheader("👞 Klasifikasi Alas Kaki (Boot/Sandal/Shoe)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔎 Mengecek apakah gambar mengandung manusia..."):
            human_detected = contains_human(img, conf_threshold=conf_threshold)

        if human_detected:
            st.error("🚫 Gambar mengandung manusia! Klasifikasi alas kaki tidak dijalankan.")
        else:
            with st.spinner("🧠 Mengklasifikasikan alas kaki..."):
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
                st.markdown("### 🛍️ Rekomendasi Produk Serupa:")

                if class_name == "Sandal":
                    st.write("- [Sandal Kulit Premium - Rp 189.000](https://tokopedia.com)")
                    st.write("- [Sandal Gunung Anti Slip - Rp 220.000](https://shopee.co.id)")
                elif class_name == "Shoe":
                    st.write("- [Sneakers Sporty X - Rp 350.000](https://tokopedia.com)")
                    st.write("- [Sepatu Formal Pria - Rp 420.000](https://shopee.co.id)")
                elif class_name == "Boot":
                    st.write("- [Boot Kulit Asli - Rp 490.000](https://tokopedia.com)")
                    st.write("- [Boot Safety Outdoor - Rp 520.000](https://shopee.co.id)")
    else:
        st.info("📤 Silakan unggah gambar atau gunakan kamera.")


# =====================================================
# STATISTIK & EKSPOR (KUSTOM WARNA)
# =====================================================
st.markdown("---")
st.markdown("### 📊 Statistik Deteksi Sementara")

col1, col2 = st.columns(2)
col1.metric("Total Deteksi Gender", st.session_state.detections["gender"])
col2.metric("Total Deteksi Alas Kaki", st.session_state.detections["footwear"])

if show_chart and len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)

    import matplotlib.pyplot as plt

    # Hitung frekuensi
    counts = df["Hasil"].value_counts()

    # Tentukan warna sesuai tema
    if theme == "🌸 Pink":
        colors = ["#FF80AB", "#F48FB1", "#EC407A"]
    elif theme == "🌲 Forest":
        colors = ["#66BB6A", "#81C784", "#A5D6A7"]
    elif theme == "🌊 Ocean":
        colors = ["#4DD0E1", "#26C6DA", "#00ACC1"]
    elif theme == "🌅 Sunset":
        colors = ["#FFB74D", "#FF8A65", "#F06292"]
    elif theme == "🌙 Dark":
        colors = ["#90CAF9", "#F48FB1", "#CE93D8"]
    else:  # Default
        colors = ["#64B5F6", "#4FC3F7", "#81D4FA"]

    # Buat plot
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
