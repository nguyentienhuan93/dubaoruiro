# 🛡️ Ứng Dụng Web Phát Hiện Giao Dịch Gian Lận (Streamlit App)

Ứng dụng này được chuyển đổi tự động từ quy trình huấn luyện mô hình học máy (`.ipynb`) sang giao diện trực quan hóa và dự báo thời gian thực sử dụng framework **Streamlit**. Ứng dụng hỗ trợ các chuyên viên phân tích rủi ro tải tập dữ liệu, tinh chỉnh tham số mô hình AI, đánh giá các chỉ số kiểm định độc lập và chạy dự báo trực tiếp hoặc hàng loạt cho tệp khách hàng mới.

## ✨ Các Tính Năng Chính
1. **Cấu hình & Tải Dữ liệu linh hoạt (Sidebar):** Hỗ trợ kéo thả file dữ liệu mẫu (`.csv`, `.xlsx`), tự động tối ưu hóa tham số mô hình theo từng họ thuật toán (Random Forest Classifier, Decision Tree Classifier, Logistic Regression).
2. **Tổng quan Dữ liệu (Tab 1):** Cung cấp cái nhìn nhanh về cấu trúc ma trận dữ liệu, số hàng, số cột và tóm tắt thống kê mô tả trung vị, phân vị cho từng đặc trưng.
3. **Trực quan hóa Dữ liệu (Tab 2):** Vẽ tự động biểu đồ phân phối biến mục tiêu lớp `default` và phân phối tần suất tích hợp biểu đồ hộp (Boxplot) của các trường dữ liệu được chọn bằng thư viện tương tác **Plotly**.
4. **Kết quả Huấn luyện (Tab 3):** Tái hiện trực quan kết quả đo lường chất lượng bao gồm Báo cáo phân loại chi tiết (Classification Report: Precision, Recall, F1-Score) và Ma trận nhầm lẫn (Confusion MatrixHeatmap).
5. **Chế độ Sử dụng Mô hình thông minh (Tab 4):**
   - **Nhập trực tiếp:** Điền thông số của 1 giao dịch cụ thể với giá trị gợi ý trung vị thông minh để nhận kết quả phân tích kèm cột xác suất trực quan.
   - **Chấm điểm hàng loạt:** Tải lên một tệp danh sách nhiều giao dịch mới có cùng cấu trúc đặc trưng để trả về nhãn phân loại tự động và hỗ trợ tải xuống (`.csv`) kết quả thẩm định.

## 🛠️ Hướng Dẫn Cài Đặt và Khởi Chạy

### 1. Cài đặt các thư viện phụ thuộc
Đảm bảo hệ thống của bạn đã cài đặt Python (Khuyến nghị phiên bản từ `3.9` đến `3.12`). Mở Terminal/Command Prompt tại thư mục chứa mã nguồn ứng dụng và chạy lệnh:
```bash
pip install -r requirements.txt
