# Tóm tắt quá trình xử lý và phân tích ICOR (1990-2024)

Tài liệu này tóm tắt các bước xử lý dữ liệu và phương pháp phân tích đã được thực hiện trong file `icor_analysis.ipynb`.

## 1. Dữ liệu đầu vào (Data Ingestion)
- **Nguồn:** Sử dụng dữ liệu từ World Bank (CSV) cho các chỉ số:
    - GCF (Gross Capital Formation - Tổng đầu tư toàn xã hội)
    - GDP Growth (Tăng trưởng GDP hàng năm)
    - FDI, Trade, Lãi suất, M2 (Dùng cho mô hình hồi quy OLS)
- **Kỹ thuật:** Xây dựng hàm `read_wb_csv` để tự động hóa việc đọc dữ liệu, bỏ qua các dòng tiêu đề thừa và xử lý cấu trúc cột của World Bank.

## 2. Tiền xử lý dữ liệu (Preprocessing)
- **Nội suy (Interpolation):** Sử dụng phương pháp nội suy tuyến tính để xử lý các giá trị thiếu (missing values), đặc biệt là dữ liệu GCF của Việt Nam trong giai đoạn đầu thập niên 90.
- **Chuẩn hóa:** Chuyển đổi định dạng dữ liệu từ dạng bảng rộng (năm là cột) sang dạng bảng dọc (năm là dòng) để phục vụ phân tích chuỗi thời gian.

## 3. Tính toán chỉ số ICOR
- **Công thức:** $ICOR = \frac{GCF (\% GDP)}{GDP Growth (\%)}$
- **Nguyên tắc:** Sử dụng **chuỗi ICOR gốc** và **không** áp dụng moving average / smoothing.

## 4. Phân tích hồi quy (OLS Regression)
- Xây dựng mô hình OLS để đánh giá tác động của các yếu tố vĩ mô đến tăng trưởng GDP.
- Kiểm định các giả thiết của mô hình (Đa cộng tuyến - VIF, Tự tương quan - Durbin-Watson).

## 5. So sánh quốc tế (Benchmarking)
- **Đối tượng:** Việt Nam so với Hàn Quốc, Trung Quốc, Thái Lan, Indonesia, Malaysia, Philippines.
- **Phương pháp:**
    - So sánh ICOR trung bình dài hạn (1990-2023).
    - So sánh ICOR 10 năm gần đây (2014-2023) để thấy xu hướng dịch chuyển hiệu quả vốn.
    - Trực quan hóa bằng biểu đồ đường (Time-series) và biểu đồ cột (Bar chart).

## 6. Các cải tiến kỹ thuật trong Notebook
- **Quản lý lỗi:** Thêm các khối `try-except` khi tính toán để đảm bảo notebook chạy thông suốt ngay cả khi dữ liệu một quốc gia bị thiếu.
- **Trực quan hóa chuyên nghiệp:** Sử dụng thư viện `matplotlib` và `seaborn` với cấu hình font tiếng Việt và phong cách hiện đại.
- **Cấu trúc:** Notebook được tổ chức thành 12 phần rõ ràng từ nhập liệu đến báo cáo kết luận.

## 7. Phân tích Chẩn đoán (Diagnostic Analytics)
- **Chow Test**: Kiểm định xem cấu trúc nền kinh tế có bị thay đổi (gãy khúc) sau hai cú sốc lớn: Khủng hoảng tài chính 2008-2009 và Đại dịch COVID-19 2020-2021.
- **T-test**: Kiểm định sự khác biệt về trung bình ICOR giữa Việt Nam và một quốc gia so sánh (Thái Lan) để xem sự khác biệt là ngẫu nhiên hay có ý nghĩa thống kê.
- **Correlation Test**: Kiểm định Pearson (Tương quan) giữa các biến số vĩ mô (FDI, M2, Lãi suất, Thương mại) với ICOR.

## 8. Mô hình ARIMAX và Phân tích Dự đoán (Predictive Analytics)
- **Kiểm định ADF (Augmented Dickey-Fuller Test)**: Kiểm tra tính dừng của chuỗi thời gian ICOR trước khi đưa vào mô hình để quyết định cần lấy sai phân hay không.
- **Biến ngoại sinh:** `macro_shock` = 1 cho các năm có |Z-score(ICOR)| > 2.5; ngược lại 0.
- **Đánh giá:** Tách **test = 3 năm cuối**; in RMSE, MAPE và p-value Ljung-Box.
- **Dự báo:** Forecast 5 năm tới với CI 95% và giả định `macro_shock=0` trong tương lai.

---
**Ghi chú:** Một số năm có thể xuất hiện ICOR rất lớn/âm khi GDP growth nhỏ hoặc đổi dấu; notebook hiện giữ cách tính ICOR gốc để phản ánh đúng dữ liệu đầu vào.
