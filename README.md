# 🦠 Dự báo lan truyền dịch bệnh từ dữ liệu di chuyển và tin tức

## 📌 Mô tả đề tài
Dự án hướng đến việc **dự báo mức độ lan truyền của dịch bệnh theo thời gian** bằng cách khai thác hai nguồn dữ liệu quan trọng: **tin tức trực tuyến** và **dữ liệu di chuyển dân cư**, với tập trung vào các bệnh như **H1N1, Ebola, Zika, MERS, Cholera**.

Bài toán được xây dựng dưới dạng **phân loại chuỗi thời gian đa bước** (multi-step classification), phân chia mức độ lan truyền thành 3 nhãn: **Thấp (0), Trung bình (1), Cao (2)**.

---

## 📚 Dữ liệu sử dụng
- 📥 **Nguồn dữ liệu**: [Time Series(Kaggle)](https://www.kaggle.com/datasets/maihongtng/time-series)
- 💾 Bao gồm các đặc trưng như: `date`, `count`, `ratio`, `label`, `count_scaled`, cùng với thông tin từ bài đăng mạng xã hội Twitter.
- 🔄 Được xử lý thống nhất qua:
  - Xử lý datetime, chuẩn hóa (`StandardScaler`, `MinMaxScaler`)
  - Chia nhãn theo phân vị (quantile)
  - Áp dụng **cửa sổ trượt** để tạo chuỗi đầu vào
  - Tách dữ liệu theo từng bệnh để huấn luyện riêng biệt

---

## 🧠 Mô hình triển khai
Ba mô hình học sâu được so sánh:

### 1. TCN (Temporal Convolutional Network)
- Dùng tích chập nhân quả giãn cách (dilated causal conv)
- Khả năng học quan hệ dài hạn
- Phù hợp với chuỗi có tính chu kỳ

### 2. PatchTST
- Cắt chuỗi thành các patch cố định
- Áp dụng Self-Attention theo từng biến
- Nhẹ, nhanh, hiệu quả với chuỗi dài

### 3. Informer
- Cải tiến từ Transformer
- Dùng **ProbSparse Attention** (O(LlogL))
- Phù hợp cho chuỗi dài, phức tạp, nhiễu

---

## ⚙️ Thực nghiệm & Đánh giá
- ✅ Huấn luyện trên 5 bệnh dịch: **Cholera, Ebola, H1N1, MERS, Zika**
- 📊 Sử dụng các chỉ số: `Accuracy`, `F1-score`, `Loss`
- 🎯 Kết quả nổi bật:
  - TCN tốt nhất với H1N1 (Acc = 92.81%)
  - PatchTST tốt với Ebola (Acc = 83.66%)
  - Informer ổn định, mạnh với MERS, Zika (Acc ≈ 85%)

---

## 💻 Giao diện ứng dụng
Triển khai giao diện người dùng tại:  
🔗 [https://huggingface.co/spaces/thanh210224/dubaolantruyenbenh](https://huggingface.co/spaces/thanh210224/dubaolantruyenbenh)

Chức năng:
- Chọn bệnh cần dự báo
- Tự động nạp mô hình đã huấn luyện
- Hiển thị dự báo & biểu đồ mức độ lan truyền

---

## 📁 Cấu trúc thư mục
├── data/ # Dữ liệu đầu vào và sau xử lý
├── models/ # Định nghĩa mô hình (TCN, PatchTST, Informer)
├── notebook/ # Notebook huấn luyện & đánh giá mô hình
├── app.py # Giao diện Gradio (triển khai trên HuggingFace)
├── utils/ # Các hàm hỗ trợ: preprocessing, metrics, attention,...
├── README.md # File mô tả dự án

## 🔧 Cài đặt

```bash
git clone https://github.com/thanhptks212k4/Du_bao_lan_truyen_benh_dua_tren_du_lieu_di_chuyen_va_tin_tuc_10gb.git
cd Du_bao_lan_truyen_benh_dua_tren_du_lieu_di_chuyen_va_tin_tuc_10gb
pip install -r requirements.txt

🧪 Hướng phát triển tương lai
Tự động hóa lựa chọn siêu tham số

Xử lý mất cân bằng nhãn

Dự báo thời gian thực và mở rộng cho nhiều vùng địa lý khác nhau

Kết hợp thêm dữ liệu dịch tễ học, thời tiết, môi trường

👨‍💻 Nhóm thực hiện
Phạm Tiến Thành – 2251262645

Chu Đức Hoàng – 2251262604

Mai Hoàng Tùng – 2251262656

Lê Hồng Nhật – 2251262623

Trường Đại học Thủy Lợi – Khoa Công nghệ thông tin
Lớp 64TTNT1 – Năm học 2024–2025
