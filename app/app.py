import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Thông số ảnh ---
IMG_HEIGHT = 180  # bạn thay bằng kích thước đúng nếu khác
IMG_WIDTH = 180

# --- Nạp mô hình ---
model = load_model("D:/FileMonHoc/Do_an_chuyen_nganh_KHDL/Bo_du_lieu/app/model/model3.keras")  

# --- Tên các lớp ---
class_names = ["BÌNH THƯỜNG (NORMAL)", "VIÊM PHỔI (PNEUMONIA)"]

# --- Hàm dự đoán ---
def predict(image):
    try:
        # Resize ảnh
        image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
        input_tensor = tf.expand_dims(np.array(image_resized), axis=0)

        # Dự đoán
        predictions = model.predict(input_tensor)
        score = predictions[0][0]

        # Phân lớp
        pred_class = 1 if score > 0.5 else 0
        label = class_names[pred_class]
        confidence = score * 100 if pred_class == 1 else (1 - score) * 100
        result_text = f"{label} 🔎 (Độ tin cậy: {confidence:.2f}%)"

        return result_text, image_resized

    except Exception as e:
        return f"Lỗi: {e}", image

# --- Giao diện Gradio ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="🖼️ Tải ảnh X-quang ngực lên"),
    outputs=[
        gr.Textbox(label="📋 Kết quả chẩn đoán"),
        gr.Image(label="Ảnh đã xử lý (Chuẩn hóa)")
    ],
    title="🩺 Hệ thống Chẩn đoán Viêm Phổi từ Ảnh X-quang",
    description="""
    Hệ thống sử dụng mô hình học sâu (Deep Learning) để phân tích ảnh X-quang ngực và dự đoán khả năng mắc **Viêm Phổi**.
    
    Vui lòng tải ảnh X-quang có định dạng JPG/PNG để kiểm tra.
    """,
    theme="soft",
    allow_flagging="never",
    live=False
)

demo.launch()
