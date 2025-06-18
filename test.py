import cv2
import numpy as np
from tkinter import Tk, filedialog, messagebox, simpledialog

# 模型加载
prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

# 人员检测函数（图片）
def detect_people_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("图像加载失败")
        return
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    person_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if class_names[idx] == "person":
                person_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"图像中人数：{person_count}")
    cv2.imshow("图像检测结果", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 视频检测函数
def detect_people_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        person_count = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if class_names[idx] == "person":
                    person_count += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

        cv2.putText(frame, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("视频检测中", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 选择操作：图片 or 视频
def choose_and_run():
    root = Tk()
    root.withdraw()

    choice = simpledialog.askstring("选择操作", "输入 '1' 选择图片检测，输入 '2' 选择视频检测：")
    if choice == '1':
        file_path = filedialog.askopenfilename(title="选择图片",
                                               filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if file_path:
            detect_people_in_image(file_path)
    elif choice == '2':
        file_path = filedialog.askopenfilename(title="选择视频",
                                               filetypes=[("Videos", "*.mp4;*.avi;*.mov")])
        if file_path:
            detect_people_in_video(file_path)
    else:
        messagebox.showinfo("提示", "无效的选择，请输入 1 或 2")

if __name__ == "__main__":
    choose_and_run()
