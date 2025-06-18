import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms
from model import LivenessModel
import time
import os
from scipy.spatial import distance as dist

# 设备和模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LivenessModel().to(device)
model.load_state_dict(torch.load("checkpoints/liveness_model.pth", map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# dlib 初始化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

# 活体动作检测阈值定义
EYE_AR_THRESH = 0.2         # 眨眼阈值
EYE_AR_CONSEC_FRAMES = 3    # 眨眼连续帧数
MAR_THRESH = 0.5            # 张嘴阈值
MOUTH_AR_CONSEC_FRAMES = 3  # 张嘴连续帧数
NOD_THRESH = 0.03           # 点头阈值（垂直移动比例）
NOD_CONSEC_FRAMES = 5       # 点头连续帧数
SHAKE_THRESH = 0.03         # 摇头阈值（水平移动比例）
SHAKE_CONSEC_FRAMES = 5     # 摇头连续帧数

# 初始化计数器
eye_counter = 0
eye_total = 0
mouth_counter = 0
mouth_total = 0
nod_counter = 0
nod_total = 0
shake_counter = 0
shake_total = 0

prev_nose = None
first_detection = True
liveness_passed = False

def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def mouth_aspect_ratio(mouth_points):
    A = dist.euclidean(mouth_points[2], mouth_points[9])
    B = dist.euclidean(mouth_points[4], mouth_points[7])
    C = dist.euclidean(mouth_points[0], mouth_points[6])
    mar = (A + B) / (2.0 * C + 1e-6)
    return mar

def nod_aspect_ratio(frame_size, prev_point, current_point):
    if prev_point is None:
        return 0
    return abs((prev_point[1] - current_point[1]) / (frame_size[0] / 2))

def shake_aspect_ratio(frame_size, prev_point, current_point):
    if prev_point is None:
        return 0
    return abs((prev_point[0] - current_point[0]) / (frame_size[1] / 2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 1:
        face = faces[0]
        landmarks = predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # 眼睛、嘴巴坐标索引
        left_eye_pts = landmarks_points[42:48]
        right_eye_pts = landmarks_points[36:42]
        mouth_pts = landmarks_points[48:68]
        nose_pts = landmarks_points[27:36]
        nose_center = np.mean(nose_pts, axis=0)

        frame_size = frame.shape

        # 计算 EAR 和 MAR
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth_pts)

        # 点头和摇头幅度
        nod_value = nod_aspect_ratio(frame_size, prev_nose, nose_center)
        shake_value = shake_aspect_ratio(frame_size, prev_nose, nose_center)
        prev_nose = nose_center

        # 眨眼检测
        if ear < EYE_AR_THRESH:
            eye_counter += 1
        else:
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                eye_total += 1
            eye_counter = 0

        # 张嘴检测
        if mar > MAR_THRESH:
            mouth_counter += 1
        else:
            if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                mouth_total += 1
            mouth_counter = 0

        # 点头检测
        if nod_value > NOD_THRESH:
            nod_counter += 1
        else:
            if nod_counter >= NOD_CONSEC_FRAMES:
                nod_total += 1
            nod_counter = 0

        # 摇头检测
        if shake_value > SHAKE_THRESH:
            shake_counter += 1
        else:
            if shake_counter >= SHAKE_CONSEC_FRAMES:
                shake_total += 1
            shake_counter = 0

        # 用深度模型判断活体
        x1, y1 = max(face.left(), 0), max(face.top(), 0)
        x2, y2 = min(face.right(), frame.shape[1]), min(face.bottom(), frame.shape[0])
        face_crop = frame[y1:y2, x1:x2]

        deep_liveness_result = False
        try:
            input_tensor = preprocess(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            deep_liveness_result = (pred == 1)
        except Exception as e:
            print("模型推理失败:", e)

        # 总结动作检测成功数量
        movement_count = sum([eye_total > 0, mouth_total > 0, nod_total > 0, shake_total > 0])

        # 判断是否活体
        if movement_count >= 2 and deep_liveness_result and first_detection:
            cv2.putText(frame, "Live detection succeed!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("活体检测通过")
            first_detection = False
            liveness_passed = True

        # 显示动作检测状态
        cv2.putText(frame, f"Eye blinks: {eye_total}", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Mouth moves: {mouth_total}", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Nod count: {nod_total}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Shake count: {shake_total}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 模型预测显示
        label = "Real" if deep_liveness_result else "Spoof"
        color = (0, 255, 0) if deep_liveness_result else (0, 0, 255)
        cv2.putText(frame, f"Model: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        # 重置计数器和状态
        eye_counter = mouth_counter = nod_counter = shake_counter = 0
        eye_total = mouth_total = nod_total = shake_total = 0
        prev_nose = None
        first_detection = True
        liveness_passed = False
        cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
