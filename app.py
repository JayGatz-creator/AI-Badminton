import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from tempfile import NamedTemporaryFile

st.title("AI Phân tích kỹ thuật cầu lông")

# Tải model đã huấn luyện
model = joblib.load("pose_model.pkl")  # Đặt file này cùng thư mục khi upload lên GitHub

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def compute_angle(a, b, c):
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

uploaded_file = st.file_uploader("Chọn video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark
            try:
                A = np.array([lm[11].x * w, lm[11].y * h])
                B = np.array([lm[13].x * w, lm[13].y * h])
                C = np.array([lm[15].x * w, lm[15].y * h])
                D = np.array([lm[12].x * w, lm[12].y * h])
                E = np.array([lm[23].x * w, lm[23].y * h])
                angle_elbow = compute_angle(A, B, C)
                shoulder_twist = compute_angle(A, D, E)
                X_new = np.array([[angle_elbow, shoulder_twist]])
                prediction = model.predict(X_new)[0]
                label = "ĐÚNG" if prediction == 1 else "SAI"
                color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
                cv2.putText(frame, f'Elbow: {angle_elbow:.1f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f'Shoulder: {shoulder_twist:.1f}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f'AI Đánh Giá: {label}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            except:
                pass
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    cap.release()
    st.success("Hoàn thành phân tích video!")