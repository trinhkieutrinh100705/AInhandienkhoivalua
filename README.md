
# 🔥 AI NHẬN DIỆN KHÓI VÀ LỬA

<p align="center">
  <a href="[https://your-link-here.com](https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/logoDaiNam.png?raw=true)">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/logoDaiNam.png width="200" style =" max-width: 100%;">
  </a>
    <a href="[https://your-link-here.com](https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/logoDaiNam.png?raw=true)">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/LogoAIoTLab.png width="170" style =" max-width: 100%" margin-left:"30">
</p>
<p align = "center">
  </a>
    <a href="https://www.facebook.com/DNUAIoTLab">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/facebook.svg >
  </a>

   </a>
    <a href="https://fitdnu.net/">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/fit.svg>
  </a>

   </a>
    <a href="https://dainam.edu.vn/vi">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/dainam.svg>
  </a>
</p>



## 📌 Giới thiệu  
Dự án này sử dụng công nghệ AI để nhận diện khói và lửa trong hình ảnh hoặc video theo thời gian thực. Bằng cách ứng dụng Deep Learning và xử lý ảnh, hệ thống có thể phát hiện nguy cơ cháy nổ một cách nhanh chóng và chính xác, giúp nâng cao khả năng cảnh báo cháy tự động.  

---
## 🏗️ HỆ THỐNG
<p align="center">
  <a href="https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/BangMach.webp">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/heThong.webp width = "70%">
  </a>

## 🚀 Tính năng  
✅ **Phát hiện khói và lửa** trong hình ảnh và video.  
✅ **Cảnh báo sớm** khi có dấu hiệu cháy nổ.  
✅ **Chạy trên GPU** để xử lý nhanh hơn.  
✅ **Tích hợp dễ dàng** vào hệ thống giám sát an ninh.  
✅ **Hỗ trợ nhiều định dạng dữ liệu**, bao gồm camera trực tiếp và video lưu trữ.  
✅ **Gửi cảnh báo qua email/SMS hoặc điều khiển thiết bị IoT.**  

---

## 🛠️ Cài đặt  
### Yêu cầu hệ thống  
- Python 3.8+  
- Hệ điều hành: Windows/Linux/macOS  
- GPU (khuyến nghị) để tăng tốc xử lý  

---

📂 Cấu trúc dự án
📞 TriTueNhanTao
├── 📂 img # Thư mục chứa hình ảnh liên quan đến hệ thống
├── nhom1.pptx # File PowerPoint trình bày về hệ thống
├── opencv_test.py # Mã nguồn sử dụng OpenCV để quét QR
├── Untitled.wfp # File dự án Wondershare Filmora
├── video.mp4 # Video minh họa hoạt động của hệ thống

🛠️ CÔNG NGHỆ Sử DỤNG

🐬 Phần cứng
Arduino LED Buzzer WiFi Camera

🖥️ Phần mềm
Python MongoDB Flask Tkinter OpenCV

🛠️ Yêu cầu hệ thống

🔌 Phần cứng
- Camera giám sát hoặc webcam
- Arduino Uno với LED và còi cảnh báo
- Cáp USB kết nối Arduino
- Raspberry Pi hoặc máy tính chạy server

💻 Phần mềm
- 🐍 Python 3+
- 🌳 MongoDB (kết nối mặc định: mongodb://localhost:27017/)
- ⚡ Arduino IDE để nạp file AlarmSystem.ino lên board Arduino.

📆 Cài đặt các thư viện Python cần thiết:
```sh
pip install pillow qrcode pymongo flask pyserial gtts pygame opencv-python
```

🫠 Bảng mạch
<p align="center">
  <a href="https://raw.githubusercontent.com/tamlehung05/NhanDienKhoiLua/refs/heads/main/img/BangMach.webp">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/BangMach.webp width = "70%">
  </a>

🛠️ Kết nối phần cứng:

| Thiết bị  | Chân kết nối | Kết nối Arduino UNO | Ghi chú |
|-------------|---------------|--------------------|----------|
| LED xanh | Anode (+), Cathode (-) | Anode → Digital Pin 9, Cathode → GND | Báo xanh khi an toàn |
| LED đỏ | Anode (+), Cathode (-) | Anode → Digital Pin 10, Cathode → GND | Báo đỏ khi phát hiện khói/lửa |
| Buzzer | (+), (-) | (+) → Digital Pin 11, (-) → GND | Còi cảnh báo |


🚀 Hướng dẫn cài đặt và chạy

1️⃣ Chuẩn bị phần cứng

Nạp mã Arduino:

Viết hoặc chỉnh sửa mã điều khiển Arduino.

Mở Arduino IDE và tải mã lên board.

Đảm bảo Arduino xuất hiện trên cổng COM5 (hoặc thay đổi trong mã nguồn Python nếu cần).

2️⃣ Cài đặt thư viện Python.

Cài đặt Python 3 nếu chưa có, sau đó cài đặt các thư viện cần thiết bằng pip.

3️⃣ Cấu hình MongoDB

Cài đặt MongoDB nếu chưa có.

Khởi động MongoDB và đảm bảo đang hoạt động tại mongodb://localhost:27017/.

4️⃣ Chạy các chương trình

✅ Chạy chương trình quét QR (opencv_test.py):

✅ Mở file trình bày (nhom1.pptx) để tham khảo chi tiết về hệ thống.

📖 Hướng dẫn sử dụng

1️⃣ Điểm danh qua QR code

Sinh viên nhận email chứa mã QR.

Khi quét mã, trình duyệt sẽ gửi yêu cầu điểm danh đến Flask server.

Hệ thống kiểm tra tính hợp lệ và cập nhật vào MongoDB, đồng thời điều khiển Arduino:

✅ Điểm danh đúng hạn → LED xanh.

⏳ Điểm danh trễ → LED đỏ, còi, phát thông báo.

2️⃣ Quản lý sinh viên & mã QR

Qua hệ thống, bạn có thể:

Thêm, sửa, xóa thông tin sinh viên.

Nhập/xuất danh sách sinh viên từ/đến file CSV.

Tạo QR cho sinh viên theo lớp hoặc toàn bộ sinh viên.

Xóa mã QR cũ một cách thủ công.

3️⃣ Xem lịch sử điểm danh

Bạn có thể:

Lọc danh sách điểm danh theo ngày, lớp, trạng thái.

Xuất dữ liệu điểm danh ra file CSV.

Hệ thống tự động cập nhật và chốt các phiên điểm danh.

⚙️ Cấu hình & Ghi chú

Cổng Arduino:

Mặc định sử dụng COM5, có thể cập nhật trong mã nguồn Python.

Email gửi mã QR:

Cập nhật thông tin sender_email và sender_password trong hệ thống.

Thời gian hiệu lực mã QR:

Mã QR có hiệu lực 100 phút kể từ thời điểm tạo.

Môi trường mạng:

Thiết bị quét QR cần kết nối cùng mạng với máy chủ.

## 📰 Poster
<p align="center">
  <a href="https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/poster.webp">
    <img src=https://github.com/tamlehung05/NhanDienKhoiLua/blob/main/img/poster.webp width = "70%">
  </a>

## 🤝 Đóng góp  

Dự án được phát triển bởi 4 thành viên:  

| Họ và Tên         | Vai trò |
|------------------|----------------------------------------------------------------|
| **Lê Hưng Tâm**   | Phát triển toàn bộ mã nguồn, thiết kế cơ sở dữ liệu, kiểm thử, triển khai dự án và thực hiện video giới thiệu. |
| **Trịnh Kiều Trinh** | Biên soạn tài liệu Overleaf, Poster, PowerPoint, thuyết trình, đề xuất cải tiến, và hỗ trợ bài tập lớn. |
| **Bùi Huy Phúc**   | Thiết kế slide PowerPoint, hỗ trợ bài tập lớn. |
| **Lê Quang Huy**   | Hỗ trợ bài tập lớn. |

© 2025 NHÓM 1, CNTT17-11 TRƯỜNG ĐẠI HỌC ĐẠI NAM  

---
© 2025 **GitHub, Inc.**  
[Terms](#)  [Privacy](#)  [Security](#)  [Status](#)  [Docs](#)  [Contact](#)  [Manage cookies](#)  [Do not share my personal information](#)




