# Phân tích chi tiết mục 14: Độ trễ tối ưu cho ICOR Việt Nam

File này diễn giải kết quả của mục **14. Phân tích độ trễ tối ưu cho ICOR** trong notebook `icor_analysis.ipynb`.

Mục tiêu của phần này là kiểm tra xem khi tính ICOR có nên dùng đầu tư cùng kỳ `GCF(t)` hay đầu tư trễ `GCF(t-k)` hay không. Về mặt kinh tế, đầu tư có thể cần thời gian để phát huy tác động vào tăng trưởng GDP, nên việc thử các lag từ 0 đến 5 năm là hợp lý.

Dữ liệu được dùng:

- GCF (% GDP): đại diện cho tỷ lệ đầu tư.
- GDP growth (%): tăng trưởng GDP hằng năm.
- Giai đoạn: 1990-2024.
- Số quan sát: 35 năm.
- Lag kiểm tra: `k = 0, 1, 2, 3, 4, 5`.

Lưu ý: kết quả cần đọc thận trọng vì mẫu chỉ có 35 quan sát, trong đó một vài năm như 2020-2022 có biến động rất lớn do COVID-19 và phục hồi sau dịch.

## 1. Cross-correlation giữa GCF(t-k) và Delta GDP(t)

### Mục đích

Phần này kiểm tra xem đầu tư ở năm hiện tại hoặc các năm trước có liên hệ với biến động tăng trưởng GDP năm hiện tại hay không.

Công thức ý tưởng:

```text
GCF(t-k)  vs  Delta GDP(t)
```

Trong đó:

- `Delta GDP(t) = GDP_growth(t) - GDP_growth(t-1)`.
- `k = 0` nghĩa là đầu tư cùng năm.
- `k = 1` nghĩa là đầu tư trễ 1 năm.
- `k = 5` nghĩa là đầu tư trễ 5 năm.

### Kết quả

| Lag k | Số quan sát | Pearson r | p-value |
|---:|---:|---:|---:|
| 0 | 34 | -0.0361 | 0.8392 |
| 1 | 33 | -0.0594 | 0.7428 |
| 2 | 32 | -0.0048 | 0.9792 |
| 3 | 31 | 0.0302 | 0.8717 |
| 4 | 30 | 0.0379 | 0.8425 |
| 5 | 29 | 0.0388 | 0.8416 |

### Diễn giải

Tất cả các hệ số tương quan đều rất gần 0. Lag có trị tuyệt đối lớn nhất là `k = 1`, nhưng `r = -0.0594`, đây là mức rất yếu. p-value của tất cả lag đều lớn hơn 0.10, nên không có lag nào có ý nghĩa thống kê.

Điều này cho thấy trong mẫu 1990-2024, **GCF(t-k) không có tương quan tuyến tính rõ ràng với biến động tăng trưởng GDP năm t**.

Kết quả này không có nghĩa là đầu tư không quan trọng. Nó chỉ cho thấy riêng chuỗi `GCF (% GDP)` không giải thích tốt biến động ngắn hạn của `GDP growth`. Tăng trưởng GDP còn bị ảnh hưởng bởi nhiều yếu tố khác như năng suất, cơ cấu ngành, xuất khẩu, tiêu dùng, chính sách tiền tệ, khủng hoảng tài chính và COVID-19.

### Kết luận cho phần 1

Không nên dựa vào Cross-correlation để khẳng định rằng đầu tư tác động mạnh nhất sau 1, 2 hay 3 năm. Bằng chứng thống kê ở phần này yếu.

Câu viết phù hợp cho báo cáo:

```text
Kết quả cross-correlation cho thấy không có lag nào từ 0 đến 5 năm đạt ý nghĩa thống kê. Hệ số tương quan đều gần 0, hàm ý rằng mối liên hệ tuyến tính giữa GCF(t-k) và biến động GDP growth trong mẫu này là rất yếu.
```

## 2. So sánh ICOR theo từng lag

### Mục đích

Phần này tính ICOR theo nhiều giả định độ trễ:

```text
ICOR(t, lag_k) = GCF(t-k) / GDP_growth(t)
```

Nếu đầu tư thực sự cần độ trễ để phát huy tác dụng, một phiên bản ICOR có lag có thể ổn định và hợp lý hơn ICOR cùng kỳ.

Tiêu chí so sánh:

- `mean`: ICOR trung bình.
- `median`: ICOR trung vị.
- `CV%`: hệ số biến thiên, càng thấp thì chuỗi càng ổn định.
- `%_in_3-6`: tỷ lệ quan sát nằm trong vùng ICOR hợp lý `[3, 6]`.
- `score`: điểm tham khảo, bằng `%_in_3-6 / CV%`; cao hơn thì tốt hơn theo quy tắc này.

### Kết quả

| Lag k | Số quan sát | Mean | Median | Std | CV% | Min | Max | Số năm trong [3,6] | % trong [3,6] | Score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 35 | 5.141 | 4.742 | 1.982 | 38.562 | 2.845 | 12.881 | 29 | 82.857 | 2.149 |
| 1 | 34 | 5.122 | 4.743 | 2.000 | 39.045 | 2.845 | 12.498 | 26 | 76.471 | 1.959 |
| 2 | 33 | 5.126 | 4.627 | 2.059 | 40.170 | 2.845 | 12.523 | 25 | 75.758 | 1.886 |
| 3 | 32 | 5.165 | 4.578 | 2.073 | 40.133 | 2.845 | 12.538 | 23 | 71.875 | 1.791 |
| 4 | 31 | 5.196 | 4.596 | 2.087 | 40.172 | 2.845 | 12.650 | 21 | 67.742 | 1.686 |
| 5 | 30 | 5.246 | 4.599 | 2.096 | 39.948 | 2.845 | 12.423 | 21 | 70.000 | 1.752 |

### Diễn giải

Kết quả này cho thấy **lag 0 là phiên bản tốt nhất theo các tiêu chí ổn định và hợp lý**:

- Lag 0 có `CV% = 38.562`, thấp nhất trong các lag.
- Lag 0 có `82.857%` quan sát nằm trong vùng `[3, 6]`, cao nhất trong các lag.
- Lag 0 có `score = 2.149`, cao nhất trong bảng.

Khi tăng lag từ 1 đến 5:

- Số quan sát giảm dần vì mất dữ liệu ở đầu chuỗi.
- `CV%` không giảm, thực tế còn tăng nhẹ lên quanh 39-40%.
- Tỷ lệ nằm trong vùng `[3, 6]` giảm từ `82.857%` ở lag 0 xuống còn `67.742%` ở lag 4 và `70.000%` ở lag 5.

Điều này hàm ý rằng việc đưa GCF về các năm trước **không làm ICOR ổn định hơn** trong bộ dữ liệu này. Nếu mục tiêu là chọn công thức ICOR để trình bày chính, ICOR cùng kỳ vẫn là lựa chọn hợp lý nhất.

### Vì sao ICOR có max cao?

Giá trị max của ICOR lag 0 là `12.881`, cao hơn nhiều so với vùng hợp lý `[3, 6]`. Nguyên nhân chính thường đến từ năm GDP growth thấp, vì GDP growth nằm ở mẫu số của công thức:

```text
ICOR = GCF / GDP_growth
```

Khi GDP growth giảm mạnh, ICOR sẽ tăng vọt dù GCF không tăng đột biến. Đây là lý do các năm COVID-19 cần được giải thích riêng trong báo cáo, tránh diễn giải máy móc là đầu tư kém hiệu quả hoàn toàn.

### Kết luận cho phần 2

Nên dùng ICOR lag 0 làm kết quả chính. Các phiên bản lag 1-5 nên được xem là kiểm tra độ nhạy, không phải phiên bản thay thế tốt hơn.

Câu viết phù hợp cho báo cáo:

```text
Khi so sánh ICOR theo các giả định độ trễ từ 0 đến 5 năm, ICOR cùng kỳ cho kết quả ổn định nhất với CV thấp nhất và tỷ lệ quan sát nằm trong vùng [3,6] cao nhất. Do đó, nghiên cứu giữ ICOR cùng kỳ làm chỉ tiêu chính và chỉ sử dụng ICOR có lag như một kiểm tra độ nhạy.
```

## 3. Granger causality: GCF có giúp dự báo GDP growth không?

### Mục đích

Kiểm định Granger trả lời câu hỏi:

```text
GCF quá khứ có giúp dự báo GDP growth tốt hơn so với chỉ dùng GDP growth quá khứ hay không?
```

Giả thuyết:

- H0: GCF không Granger-cause GDP growth.
- Nếu p-value < 0.05: có bằng chứng bác bỏ H0 ở mức 5%.
- Nếu p-value < 0.10: có bằng chứng yếu hơn ở mức 10%.

Do chuỗi thời gian kinh tế vĩ mô có nguy cơ không dừng, phần kiểm định được diễn giải thận trọng trên chuỗi sai phân.

### Kết quả F-test trên chuỗi sai phân

| Lag | F-stat | p-value | Số quan sát |
|---:|---:|---:|---:|
| 1 | 0.0675 | 0.7968 | 33 |
| 2 | 0.0565 | 0.9452 | 32 |
| 3 | 0.0251 | 0.9945 | 31 |
| 4 | 0.0583 | 0.9932 | 30 |
| 5 | 0.1403 | 0.9805 | 29 |

### Diễn giải

Tất cả p-value đều rất lớn, lớn hơn 0.10. Lag có p-value nhỏ nhất là lag 1, nhưng `p = 0.7968`, vẫn quá cao để bác bỏ H0.

Vì vậy, kết quả Granger cho thấy:

```text
Chưa có bằng chứng thống kê rằng GCF quá khứ giúp dự báo GDP growth trong mẫu 1990-2024.
```

Điều này không phủ nhận vai trò của đầu tư trong tăng trưởng dài hạn. Nó chỉ nói rằng với dữ liệu hằng năm và cách đo bằng `GCF (% GDP)`, thông tin về GCF quá khứ không cải thiện đáng kể khả năng dự báo GDP growth theo kiểm định Granger.

Một số lý do có thể:

- Mẫu nhỏ, chỉ khoảng 35 năm.
- GCF là tỷ lệ trên GDP, không tách được đầu tư công, tư nhân, FDI, chất lượng đầu tư.
- GDP growth bị tác động mạnh bởi các cú sốc khác như khủng hoảng tài chính, COVID-19, cầu xuất khẩu, cầu nội địa.
- Tác động của đầu tư có thể phi tuyến hoặc phụ thuộc chất lượng vốn, không nằm trong quan hệ tuyến tính đơn giản.

### Kết luận cho phần 3

Không nên viết rằng GCF Granger-cause GDP growth. Cách viết đúng hơn là:

```text
Kiểm định Granger không bác bỏ giả thuyết H0 ở tất cả các lag từ 1 đến 5. Do đó, trong mẫu dữ liệu này, chưa có bằng chứng rằng GCF quá khứ có khả năng dự báo GDP growth.
```

## 4. Kết luận chung cho mục 14

Ba phần cho kết quả khá nhất quán:

1. Cross-correlation không tìm thấy lag nào có tương quan có ý nghĩa thống kê.
2. ICOR theo lag cho thấy lag 0 ổn định và hợp lý nhất.
3. Granger causality không cho thấy GCF quá khứ giúp dự báo GDP growth.

Kết luận chính:

```text
Trong bộ dữ liệu Việt Nam 1990-2024, chưa có bằng chứng thống kê đủ mạnh để khẳng định cần sử dụng độ trễ đầu tư khi tính ICOR. ICOR cùng kỳ vẫn là phiên bản phù hợp nhất để trình bày chính, còn ICOR có lag nên được dùng như phần kiểm tra độ nhạy.
```

## 5. Nên dùng kết quả này trong bài như thế nào?

### Nếu viết phần phương pháp

Có thể viết:

```text
Bên cạnh ICOR cùng kỳ, nghiên cứu kiểm tra thêm các phiên bản ICOR với độ trễ đầu tư từ 1 đến 5 năm. Mục đích là đánh giá liệu đầu tư trong quá khứ có phản ánh tốt hơn tăng trưởng GDP hiện tại hay không. Các tiêu chí so sánh gồm hệ số tương quan, độ ổn định của ICOR và kiểm định Granger causality.
```

### Nếu viết phần kết quả

Có thể viết:

```text
Kết quả cho thấy các hệ số cross-correlation giữa GCF(t-k) và Delta GDP(t) đều gần 0 và không có ý nghĩa thống kê. Khi tính ICOR theo từng lag, lag 0 có CV thấp nhất và tỷ lệ quan sát trong vùng [3,6] cao nhất. Kiểm định Granger cũng không bác bỏ giả thuyết GCF không dự báo GDP growth. Vì vậy, nghiên cứu tiếp tục sử dụng ICOR cùng kỳ làm chỉ tiêu chính.
```

### Nếu viết phần hạn chế

Có thể viết:

```text
Phân tích độ trễ chịu hạn chế bởi kích thước mẫu nhỏ và dữ liệu hằng năm. Ngoài ra, GCF (% GDP) chỉ phản ánh quy mô đầu tư tương đối, chưa phản ánh chất lượng đầu tư, cơ cấu đầu tư và độ trễ riêng của từng loại dự án. Do đó, kết quả không nên được diễn giải như bằng chứng phủ nhận mọi quan hệ giữa đầu tư và tăng trưởng.
```

## 6. Khuyến nghị trình bày

Trong báo cáo hoặc slide, nên trình bày theo hướng:

- Giữ ICOR cùng kỳ là kết quả chính.
- Đưa bảng ICOR theo lag để chứng minh đã kiểm tra độ nhạy.
- Nói rõ rằng các lag 1-5 không cải thiện tính ổn định của ICOR.
- Không khẳng định quan hệ nhân quả từ Granger test.
- Nếu cần mở rộng nghiên cứu, nên tách GCF thành đầu tư công, đầu tư tư nhân và FDI, hoặc bổ sung biến năng suất, lao động, xuất khẩu và lạm phát.
