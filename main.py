import streamlit as st
from tensorflow import keras
import cv2

# Конфигурация
MODEL_PATH = "path/to/your/model.h5"  # Путь к сохранённой модели
VIDEO_FEED_URL = "rtsp://.../"  # Путь к видеофайлу
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

# CamDetection.py возвращает frames (внутри есть данные по времени обработки)

def main():
    st.title("Детекция собак по камерам города")
    model = load_model()

    if model:
        try:
            cap = cv2.VideoCapture(VIDEO_FEED_URL)
            if not cap.isOpened():
                st.error("Не удалось открыть видеопоток")
                return
        except Exception as e:
            st.error(f"Ошибка открытия видеопотока: {e}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = CapDetection(frame, model)

            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if st.button("Остановить"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#Network URL: http://192.168.1.96:8501