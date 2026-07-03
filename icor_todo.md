# Huong dan phan tich ICOR Viet Nam

## 1) Dinh nghia va cong thuc
- ICOR (Incremental Capital-Output Ratio) phan anh hieu qua su dung von dau tu.
- Cong thuc: ICOR_t = GCF_t / GDP_growth_t (cung don vi %).
- Dien giai: ICOR thap -> hieu qua dau tu cao.

## 2) Bien va du lieu can thu thap
- GCF (% GDP): Gross Capital Formation (% of GDP).
- GDP growth (%): GDP Growth (annual %).
- Bien ho tro: FDI (% GDP), Trade (% GDP), Lending Interest Rate (%), Broad Money M2 (% GDP).
- Neu so sanh quoc gia: chon Thai Lan, Indonesia, Trung Quoc, Han Quoc (tuy nguon du lieu).

## 3) Xu ly du lieu
- Loc Viet Nam, chi giu cac nam chung giua cac chi so.
- Chuan hoa kieu so, xu ly thieu du lieu (blank, "..", NA).
- Kiem tra outlier, chu y cac nam khung hoang (2008-2009, 2020).

## 4) Tinh ICOR va phien ban do nhay
- Co ban: ICOR_t = GCF_t / GDP_growth_t.
- Lag 1 nam: ICOR_t = GCF_{t-1} / GDP_growth_t.
- Lam tron: trung binh truot 3-5 nam de thay xu huong.

## 5) Phan tich xu huong va EDA
- Ve chuoi thoi gian GCF, GDP growth, ICOR.
- Nhan dien nam bat thuong va thay doi cau truc.
- So sanh ICOR theo giai doan (pre/post 2007, 2011, 2020).

## 6) Chon mo hinh theo muc tieu
- Neu muc tieu mo ta/giai thich:
	- Phan tich xu huong + hoi quy tuyen tinh don/da bien (OLS).
	- Kiem dinh diem gay (structural break).
	- Ly do: mau nho, de dien giai.
- Neu muc tieu du bao ngan han:
	- ARIMA/ETS cho ICOR hoac GDP growth.
	- Ly do: chuoi thoi gian ngan, mo hinh don gian on dinh.
- Neu muon quan he dong giua nhieu bien:
	- VAR/VECM (neu du du lieu va co dong lien ket).
	- Ly do: mo ta tac dong qua thoi gian tot hon OLS.

## 7) Cac cau hoi nghien cuu goi y
- ICOR Viet Nam bien dong the nao giai doan 1990-2023? Nam nao cao bat thuong va vi sao?
- Dau tu cong, FDI, dau tu tu nhan anh huong ICOR khac nhau ra sao?
- ICOR Viet Nam so voi Thai Lan, Indonesia, Trung Quoc o muc nao?

## 8) Danh gia va giai thich
- Kiem tra dieu kien mo hinh (stationarity, residuals).
- So sanh cac phien ban (GCF hien tai vs GCF tre 1 nam).
- Rut ra y nghia kinh te va ham y chinh sach.

## 9) Bao cao
- Tom tat ket qua, bieu do chinh.
- Neu han che va canh bao ve du lieu.
- Ket luan ngan gon, de xuat ham y.

## 10) Nguon du lieu tham khao
- GCF: World Bank (NE.GDI.TOTL.ZS)
- GDP growth: World Bank (NY.GDP.MKTP.KD.ZG)
- FDI: World Bank (BX.KLT.DINV.WD.GD.ZS)
- Trade: World Bank (NE.TRD.GNFS.ZS)
- Lending Interest Rate: World Bank (FR.INR.LEND)
- M2: World Bank (FM.LBL.BMNY.GD.ZS)
- Dau tu cong: IMF (Investment and Capital Stock)
