# Giai thich phan tich lag cho ICOR Viet Nam

File nay giai thich phan `lag_analysis.py` da duoc tich hop vao notebook chinh `icor_analysis.ipynb`.

## 1. Muc tieu

Phan tich lag dung de kiem tra do tre hop ly giua dau tu va tang truong GDP. Ve mat kinh te, von dau tu thuong khong tao ra tang truong ngay trong cung nam, ma co the can 1-5 nam de phat huy tac dong.

Notebook kiem tra ba huong:

- Cross-correlation giua `GCF(t-k)` va `Delta GDP(t)`.
- ICOR voi tung gia dinh lag `k = 0..5`.
- Granger causality de xem GCF qua khu co giup du bao tang truong GDP hay khong.

## 2. Cross-correlation

Cong thuc so sanh:

```text
GCF(t-k)  vs  Delta GDP(t)
```

Trong do:

- `GCF` la Gross Capital Formation (% GDP), dai dien cho ty le dau tu.
- `Delta GDP` la sai phan bac 1 cua tang truong GDP.
- `k` la so nam tre. Vi du `k=2` nghia la dau tu nam `t-2` duoc so sanh voi bien dong tang truong GDP nam `t`.

Cach doc ket qua:

- `pearson_r` cang xa 0 thi tuong quan cang manh.
- `p_value < 0.05` thuong duoc xem la co y nghia thong ke manh.
- `p_value < 0.10` co the xem la bang chung yeu trong mau nho.

Luu y: tuong quan khong dong nghia voi quan he nhan qua.

## 3. ICOR theo tung lag

Cong thuc:

```text
ICOR(t, lag_k) = GCF(t-k) / GDP_growth(t)
```

Phan nay tinh ICOR cho tung `k = 0..5`, sau do so sanh:

- Trung binh va trung vi ICOR.
- Do lech chuan va he so bien thien `CV%`.
- Ty le quan sat nam trong vung hop ly `[3, 6]`.

Cach doc ket qua:

- Lag tot nen co `CV%` thap, vi ICOR on dinh hon.
- Lag tot nen co ty le nam trong `[3, 6]` cao, vi vung nay thuong hop ly voi nen kinh te dang phat trien.
- Diem `score = %_in_3-6 / CV%` chi la quy tac chon nhanh, khong phai chan ly kinh te tuyet doi.

## 4. Granger causality

Kiem dinh Granger tra loi cau hoi:

```text
Du lieu GCF qua khu co giup du bao GDP_growth hien tai/tiep theo tot hon khong?
```

Gia thuyet:

- H0: GCF khong Granger-cause GDP growth.
- Neu `p_value < 0.05`, co the bac bo H0 o muc 5%.
- Neu `p_value < 0.10`, co bang chung yeu hon o muc 10%.

Truoc khi kiem dinh, notebook dung ADF test de kiem tra tinh dung cua chuoi. Neu it nhat mot chuoi khong dung, code chuyen sang sai phan bac 1 de giam rui ro hoi quy gia.

Luu y: Granger causality la kha nang du bao theo thoi gian, khong tu dong chung minh quan he nhan qua kinh te.

## 5. File dau ra

Khi chay phan lag analysis, cac bieu do duoc luu vao thu muc:

```text
lag_analysis_output/
```

Gom:

- `01_cross_correlation.png`
- `02_icor_by_lag.png`
- `03_icor_cv_comparison.png`
- `04_granger_causality.png`

## 6. Cach dua vao bao cao

Nen trinh bay ket qua theo thu tu:

1. Lag nao co tuong quan cao nhat giua dau tu va bien dong tang truong.
2. Lag nao cho ICOR on dinh va nam nhieu trong vung `[3, 6]`.
3. Granger test co bac bo H0 hay khong.
4. Ket luan chon lag nen duoc dien giai than trong, vi mau nam 1990-2024 chi co khoang 35 quan sat.
