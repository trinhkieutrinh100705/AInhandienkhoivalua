import cv2 # type: ignore
import numpy as np # type: ignore
import os



# Mở video
cap = cv2.VideoCapture(r"D:\TriTueNhanTao\video.mp4")
if not cap.isOpened():
    print("Không thể mở video! Kiểm tra đường dẫn hoặc định dạng file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Số khung hình trên giây
delay = int(1000 / fps)  # Tính thời gian chờ để video chạy đúng tốc độ


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Xác định ngưỡng màu cho lửa (đỏ, cam, vàng)
    lower_fire = np.array([18, 50, 50], dtype=np.uint8)
    upper_fire = np.array([35, 255, 255], dtype=np.uint8)

    # Xác định ngưỡng màu cho khói (trắng, xám)
    lower_smoke = np.array([0, 0, 100], dtype=np.uint8)
    upper_smoke = np.array([180, 50, 255], dtype=np.uint8)

    # Tạo mặt nạ (mask) để phát hiện lửa và khói
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    # Hiển thị vùng phát hiện lửa và khói
    fire_detected = cv2.bitwise_and(frame, frame, mask=fire_mask)
    smoke_detected = cv2.bitwise_and(frame, frame, mask=smoke_mask)

    # Vẽ khung chữ nhật khi phát hiện lửa
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Lọc các vùng nhỏ để tránh nhiễu
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE DETECTED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Vẽ khung chữ nhật khi phát hiện khói
    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, "SMOKE DETECTED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hiển thị video
    cv2.imshow("Fire and Smoke Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
print(os.listdir())  # Kiểm tra danh sách file trong thư mục hiện tại
cap.release()
cv2.destroyAllWindows()
