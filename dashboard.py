# =====================================================
# 👟 Gender & Footwear AI Detection Dashboard (No Voice)
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
Aplikasi ini menggunakan *YOLOv8* untuk deteksi gender dan *CNN (TensorFlow)* untuk klasifikasi alas kaki.  
Model bekerja pada domainnya masing-masing:
- 🧍 *YOLO:* Men / Women  
- 👞 *CNN:* Boot / Sandal / Shoe  
""")

# =====================================================
# 🌙 DARK MODE OPSIONAL
# =====================================================
st.sidebar.markdown("### 🌙 Tema Tampilan")
dark_mode = st.sidebar.checkbox("Aktifkan Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: #FAFAFA; }
        .stApp { background-color: #0E1117; }
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

        if detections:
            valid = any(d["label"] in ["Men", "Women"] for d in detections)
            if valid:
                st.session_state.detections["gender"] += 1
                gender_detected = detections[0]["label"]
                st.session_state.history.append({"Tipe": "Gender", "Hasil": gender_detected})

                # 🔹 Rekomendasi berdasarkan gender
                if gender_detected == "Men":
                    st.markdown("### 🧴 Rekomendasi untuk Pria")
                    st.info("""
                    - Gunakan *moisturizer* harian untuk menjaga kelembapan kulit.  
                    - Pilihan outfit kasual: **kemeja polos + jeans slim fit**.  
                    - Produk rekomendasi utama: **Face Wash Men Deep Clean - Rp 35.000**
                    """)
                    st.markdown("#### 🛒 Belanja Sekarang:")
                    st.write("- [🛍️ Face Wash Men Deep Clean (Tokopedia)](https://www.tokopedia.com/search?st=product&q=face%20wash%20men%20deep%20clean)")
                    st.write("- [🧴 Moisturizer Men (Shopee)](https://shopee.co.id/search?keyword=moisturizer%20men)")
                    st.write("- [👕 Kemeja Polos Pria (Tokopedia)](https://www.tokopedia.com/search?st=product&q=kemeja%20polos%20pria)")
                    st.write("- [👟 Sepatu Kasual Pria (Shopee)](https://shopee.co.id/search?keyword=sepatu%20kasual%20pria)")
                    st.write("- [⌚ Jam Tangan Sporty (Tokopedia)](https://www.tokopedia.com/search?st=product&q=jam%20tangan%20pria)")

                elif gender_detected == "Women":
                    st.markdown("### 💅 Rekomendasi untuk Wanita")
                    st.info("""
                    - Gunakan *sunscreen* setiap hari (minimal SPF 30+) untuk melindungi kulit dari UV.  
                    - Coba gaya kasual dengan **floral dress** dan aksesori minimalis.  
                    - Produk rekomendasi utama: **Serum Vitamin C Bright - Rp 50.000**
                    """)
                    st.markdown("#### 🛒 Belanja Sekarang:")
                    st.write("- [☀️ Sunscreen SPF 30+ (Shopee)](https://shopee.co.id/search?keyword=sunscreen%20spf%2030)")
                    st.write("- [🌸 Floral Dress Casual (Tokopedia)](https://www.tokopedia.com/search?st=product&q=floral%20dress)")
                    st.write("- [💧 Serum Vitamin C Bright (Shopee)](https://shopee.co.id/search?keyword=serum%20vitamin%20c%20bright)")
                    st.write("- [👜 Tas Fashion Wanita (Tokopedia)](https://www.tokopedia.com/search?st=product&q=tas%20wanita)")
                    st.write("- [👠 High Heels Elegant (Shopee)](https://shopee.co.id/search?keyword=high%20heels%20elegant)")
            else:
                st.warning("⚠ Gambar bukan domain gender.")
        else:
            st.warning("⚠ Tidak ada objek terdeteksi.")
    else:
        st.info("📤 Silakan unggah gambar atau gunakan kamera.")

# =====================================================
# MODE: CNN (Klasifikasi Alas Kaki)
# =====================================================
elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
    st.subheader("👞 Klasifikasi Alas Kaki (Boot/Sandal/Shoe)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

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
# STATISTIK & EKSPOR
# =====================================================
st.markdown("---")
st.markdown("### 📊 Statistik Deteksi Sementara")
col1, col2 = st.columns(2)
col1.metric("Total Deteksi Gender", st.session_state.detections["gender"])
col2.metric("Total Deteksi Alas Kaki", st.session_state.detections["footwear"])

if show_chart and len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.bar_chart(df["Hasil"].value_counts())

if export_enable and len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Unduh Riwayat Deteksi (CSV)", data=csv, file_name="riwayat_deteksi.csv", mime="text/csv")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart AI Vision — Leni Gustia 👩‍💻 | YOLOv8 + TensorFlow CNN + Dark Mode + Smart Export")
