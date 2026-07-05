"""
Phân tích Lag tối ưu cho ICOR Việt Nam
========================================
Mục tiêu:
1. Cross-correlation giữa Đầu tư(t-k) và ΔGDP(t) với k = 0..MAX_LAG
2. Tính ICOR với từng giả định lag (0..MAX_LAG) → so sánh tính ổn định & hợp lý (3–6)
3. Kiểm định Granger causality: đầu tư ở lag nào "dẫn dắt" tăng trưởng GDP mạnh nhất

Dữ liệu: World Bank CSV – GCF (% GDP) và GDP growth (%)
"""
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_context("notebook")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})

# ---------- 0. CONFIG & DATA LOADING ----------
BASE_DIR = Path(__file__).parent
START_YEAR = 1990
END_YEAR = 2024
MAX_LAG = 34  # Tu sua so k toi da o day, vi du: 10, 15, 20.


def cap_max_lag(max_lag: int, n_obs: int, min_obs: int = 5, label: str = "phan tich") -> int:
    """Gioi han max lag de moi lag con du so quan sat toi thieu."""
    max_allowed = max(0, n_obs - min_obs)
    if max_lag > max_allowed:
        print(f"[Canh bao] MAX_LAG={max_lag} qua lon cho {label}; tu dong giam ve {max_allowed}.")
        return max_allowed
    return max_lag


OUTPUT_DIR = BASE_DIR / "lag_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def read_wb_csv(path: Path) -> pd.DataFrame | None:
    """Đọc CSV World Bank, bỏ 4 dòng metadata đầu."""
    if not path.exists():
        print(f"  [THIẾU] {path}")
        return None
    df = pd.read_csv(path, skiprows=4)
    print(f"  [OK] {path.name}: {df.shape}")
    return df


def to_long_vn(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide → long cho Việt Nam."""
    if df is None:
        return pd.DataFrame(columns=["year", value_name])
    data = df[df["Country Name"].str.casefold() == "viet nam"].copy()
    year_cols = [c for c in data.columns if str(c).isdigit()]
    id_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    long = data[id_cols + year_cols].melt(id_vars=id_cols, var_name="year", value_name=value_name)
    long["year"] = long["year"].astype(int)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long[["year", value_name]]


print("=" * 70)
print("BƯỚC 0: ĐỌC VÀ CHUẨN BỊ DỮ LIỆU")
print("=" * 70)

gcf_path = BASE_DIR / "GCF" / "API_NE.GDI.TOTL.ZS_DS2_en_csv_v2_115698.csv"
gdp_path = BASE_DIR / "GDP" / "API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_121708.csv"

raw_gcf = read_wb_csv(gcf_path)
raw_gdp = read_wb_csv(gdp_path)

gcf_long = to_long_vn(raw_gcf, "gcf")
gdp_long = to_long_vn(raw_gdp, "gdp_growth")

df = gcf_long.merge(gdp_long, on="year", how="inner")
df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()
df = df.sort_values("year").reset_index(drop=True)

# Nội suy nếu thiếu
df = df.set_index("year").interpolate(method="linear", limit_direction="both").reset_index()

# ΔGDP – sai phân bậc 1 của GDP growth
df["delta_gdp"] = df["gdp_growth"].diff()

print(f"\nDữ liệu sẵn sàng: {len(df)} năm ({df['year'].min()}–{df['year'].max()})")
print(df[["year", "gcf", "gdp_growth", "delta_gdp"]].head(10).to_string(index=False))
print()

MAX_LAG = cap_max_lag(
    MAX_LAG,
    n_obs=len(df.dropna(subset=["delta_gdp"])),
    min_obs=5,
    label="cross-correlation và ICOR",
)
LAG_VALUES = list(range(MAX_LAG + 1))
print(f"Max lag đang dùng: k = 0..{MAX_LAG}")

# ====================================================================
# PHẦN 1: CROSS-CORRELATION – Đầu tư(t-k) vs ΔGDP(t)
# ====================================================================
print("=" * 70)
print("PHẦN 1: CROSS-CORRELATION  –  Investment(t-k)  vs  ΔGDP(t)")
print("=" * 70)

# Lấy chuỗi hợp lệ (bỏ NaN do diff)
valid = df.dropna(subset=["delta_gdp"]).copy()

results_cc = []
for k in LAG_VALUES:
    # GCF(t-k) → lag k kỳ → dịch chuỗi GCF về trước k bước
    shifted = valid["gcf"].shift(k)
    mask = shifted.notna() & valid["delta_gdp"].notna()
    x = shifted[mask].values
    y = valid.loc[mask, "delta_gdp"].values
    n = len(x)
    if n < 5:
        continue
    r, pval = pearsonr(x, y)
    results_cc.append({"lag_k": k, "n_obs": n, "pearson_r": r, "p_value": pval})

cc_df = pd.DataFrame(results_cc)
print("\n>>> Bảng Cross-Correlation: Investment(t-k) & ΔGDP(t)")
print(cc_df.to_string(index=False))

# Xác định lag tốt nhất (|r| lớn nhất & p < 0.10)
best_cc = cc_df.loc[cc_df["pearson_r"].abs().idxmax()]
print(f"\n→ Lag có |r| lớn nhất: k = {int(best_cc['lag_k'])} "
      f"(r = {best_cc['pearson_r']:.4f}, p = {best_cc['p_value']:.4f})")
sig = cc_df[cc_df["p_value"] < 0.10]
if not sig.empty:
    best_sig = sig.loc[sig["pearson_r"].abs().idxmax()]
    print(f"→ Lag có ý nghĩa thống kê (p < 0.10) & |r| cao nhất: k = {int(best_sig['lag_k'])} "
          f"(r = {best_sig['pearson_r']:.4f}, p = {best_sig['p_value']:.4f})")
else:
    print("  (Không có lag nào có p < 0.10)")

# --- Biểu đồ cross-correlation ---
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ['#2ca02c' if p < 0.05 else ('#ff7f0e' if p < 0.10 else '#1f77b4')
          for p in cc_df["p_value"]]
bars = ax.bar(cc_df["lag_k"], cc_df["pearson_r"], color=colors, edgecolor="black", width=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag k (kỳ trễ đầu tư)")
ax.set_ylabel("Hệ số tương quan Pearson (r)")
ax.set_title("Cross-Correlation: Investment(t−k)  vs  ΔGDP(t)")
ax.set_xticks(cc_df["lag_k"].astype(int).tolist())
for i, row in cc_df.iterrows():
    ax.text(row["lag_k"], row["pearson_r"] + 0.02 * np.sign(row["pearson_r"]),
            f'{row["pearson_r"]:.3f}\np={row["p_value"]:.3f}',
            ha="center", va="bottom" if row["pearson_r"] >= 0 else "top", fontsize=8)
# Legend thủ công
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='p < 0.05'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='p < 0.10'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='p ≥ 0.10'),
]
ax.legend(handles=legend_elements, loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_cross_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[Lưu biểu đồ] {OUTPUT_DIR / '01_cross_correlation.png'}")

# ====================================================================
# PHẦN 2: ICOR THEO TỪNG LAG – SO SÁNH TÍNH ỔN ĐỊNH & HỢP LÝ
# ====================================================================
print("\n" + "=" * 70)
print(f"PHẦN 2: ICOR THEO TỪNG LAG (k=0..{MAX_LAG})  –  SO SÁNH ỔN ĐỊNH & HỢP LÝ")
print("=" * 70)

REASONABLE_LOW = 3.0
REASONABLE_HIGH = 6.0

icor_summary = []
icor_series = {}

for k in LAG_VALUES:
    col = f"icor_lag{k}"
    # ICOR(t, lag_k) = GCF(t-k) / GDP_growth(t)
    gcf_shifted = df["gcf"].shift(k)
    # Bỏ năm GDP growth <= 0 (ICOR vô nghĩa khi suy thoái)
    mask = (df["gdp_growth"] > 0) & gcf_shifted.notna()
    icor_vals = gcf_shifted[mask] / df.loc[mask, "gdp_growth"]
    years_valid = df.loc[mask, "year"]

    icor_series[k] = pd.DataFrame({"year": years_valid, "icor": icor_vals.values})

    # Thống kê
    n_in_range = ((icor_vals >= REASONABLE_LOW) & (icor_vals <= REASONABLE_HIGH)).sum()
    pct_in_range = n_in_range / len(icor_vals) * 100 if len(icor_vals) > 0 else 0
    cv = icor_vals.std() / icor_vals.mean() * 100 if icor_vals.mean() != 0 else np.nan  # CV%

    icor_summary.append({
        "lag_k": k,
        "n_obs": len(icor_vals),
        "mean": icor_vals.mean(),
        "median": icor_vals.median(),
        "std": icor_vals.std(),
        "CV%": cv,
        "min": icor_vals.min(),
        "max": icor_vals.max(),
        "n_in_3-6": int(n_in_range),
        "%_in_3-6": pct_in_range,
    })

summary_df = pd.DataFrame(icor_summary)
print("\n>>> Bảng tổng hợp ICOR theo từng Lag")
print(summary_df.to_string(index=False, float_format="%.3f"))

# Chọn lag tốt nhất: ưu tiên CV% thấp nhất (ổn định), sau đó %_in_3-6 cao nhất
summary_df["score"] = summary_df["%_in_3-6"] / (summary_df["CV%"] + 1e-6)
best_icor_lag = summary_df.loc[summary_df["score"].idxmax()]
print(f"\n→ Lag có ICOR ổn định & hợp lý nhất: k = {int(best_icor_lag['lag_k'])}")
print(f"  - Mean = {best_icor_lag['mean']:.3f}, Median = {best_icor_lag['median']:.3f}")
print(f"  - CV = {best_icor_lag['CV%']:.1f}%, % nằm trong [3–6] = {best_icor_lag['%_in_3-6']:.1f}%")

# --- Biểu đồ ICOR theo từng lag ---
n_lags = len(LAG_VALUES)
n_cols = min(3, n_lags)
n_rows = math.ceil(n_lags / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.3 * n_cols, 4.2 * n_rows), sharex=True, squeeze=False)
axes = axes.ravel()

for ax_idx, idx in enumerate(LAG_VALUES):
    ax = axes[ax_idx]
    ser = icor_series[idx]
    ax.plot(ser["year"], ser["icor"], marker="o", markersize=4, linewidth=1.3, color="teal")
    ax.axhspan(REASONABLE_LOW, REASONABLE_HIGH, alpha=0.15, color="green", label="Vùng hợp lý [3–6]")
    ax.axhline(summary_df.loc[idx, "mean"], color="red", linestyle="--", linewidth=1, label=f"Mean={summary_df.loc[idx, 'mean']:.2f}")
    ax.set_title(f"ICOR Lag {idx}  (CV={summary_df.loc[idx, 'CV%']:.1f}%)")
    ax.set_ylabel("ICOR")
    ax.set_ylim(-5, 30)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

for ax in axes[n_lags:]:
    ax.set_visible(False)

for ax in axes[max(0, n_lags - n_cols):n_lags]:
    ax.set_xlabel("Năm")

fig.suptitle(f"ICOR Việt Nam theo từng giả định Lag (k=0..{MAX_LAG})", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_icor_by_lag.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[Lưu biểu đồ] {OUTPUT_DIR / '02_icor_by_lag.png'}")

# --- Biểu đồ so sánh CV% và %_in_3-6 ---
fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = summary_df["lag_k"]
ax1.bar(x - 0.15, summary_df["CV%"], width=0.3, color="#1f77b4", label="CV (%)", edgecolor="black")
ax1.set_xlabel("Lag k")
ax1.set_ylabel("Hệ số biến thiên CV (%)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")

ax2 = ax1.twinx()
ax2.bar(x + 0.15, summary_df["%_in_3-6"], width=0.3, color="#2ca02c", label="% in [3–6]", edgecolor="black")
ax2.set_ylabel("% quan sát nằm trong [3–6]", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")

ax1.set_xticks(summary_df["lag_k"].astype(int).tolist())
ax1.set_title("So sánh tính ổn định (CV) & hợp lý (% in [3–6]) theo Lag")
# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_icor_cv_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Lưu biểu đồ] {OUTPUT_DIR / '03_icor_cv_comparison.png'}")

# ====================================================================
# PHẦN 3: GRANGER CAUSALITY – ĐẦU TƯ → TĂNG TRƯỞNG GDP
# ====================================================================
print("\n" + "=" * 70)
print("PHẦN 3: KIỂM ĐỊNH GRANGER CAUSALITY")
print("  H0: GCF KHÔNG Granger-cause GDP_growth")
print("=" * 70)

# Chuẩn bị chuỗi: kiểm tra tính dừng
print("\n--- Kiểm định ADF (tính dừng) ---")
for col_name, col_data in [("gcf", df["gcf"]), ("gdp_growth", df["gdp_growth"])]:
    adf_result = adfuller(col_data.dropna(), autolag="AIC")
    stat_label = "DỪNG ✓" if adf_result[1] < 0.05 else "KHÔNG DỪNG ✗"
    print(f"  {col_name:12s}: ADF stat = {adf_result[0]:.4f}, p = {adf_result[1]:.4f}  → {stat_label}")

# Nếu chuỗi không dừng → dùng sai phân
gcf_series = df["gcf"].values
gdp_series = df["gdp_growth"].values

# Kiểm tra tính dừng
adf_gcf = adfuller(gcf_series, autolag="AIC")
adf_gdp = adfuller(gdp_series, autolag="AIC")

use_diff = False
if adf_gcf[1] > 0.05 or adf_gdp[1] > 0.05:
    print("\n  → Ít nhất một chuỗi không dừng → Sử dụng sai phân bậc 1 cho Granger test")
    gcf_diff = np.diff(gcf_series)
    gdp_diff = np.diff(gdp_series)
    use_diff = True
    # Kiểm tra lại
    adf_gcf_d = adfuller(gcf_diff, autolag="AIC")
    adf_gdp_d = adfuller(gdp_diff, autolag="AIC")
    print(f"  Δgcf       : ADF stat = {adf_gcf_d[0]:.4f}, p = {adf_gcf_d[1]:.4f}")
    print(f"  Δgdp_growth: ADF stat = {adf_gdp_d[0]:.4f}, p = {adf_gdp_d[1]:.4f}")
else:
    print("\n  → Cả hai chuỗi đều dừng → Dùng chuỗi gốc cho Granger test")

# Tạo dữ liệu cho Granger test
if use_diff:
    granger_data = np.column_stack([gdp_diff, gcf_diff])  # [y, x] – y là biến phụ thuộc
else:
    granger_data = np.column_stack([gdp_series, gcf_series])

MAX_GRANGER_LAG = min(MAX_LAG, max(1, (len(granger_data) - 2) // 3))
if MAX_GRANGER_LAG < MAX_LAG:
    print(f"  [Cảnh báo] MAX_LAG={MAX_LAG} quá lớn cho Granger test; tự động dùng max lag = {MAX_GRANGER_LAG}.")

print(f"\n--- Granger Causality Test (max lag = {MAX_GRANGER_LAG}) ---")
print(f"  Dữ liệu: {'sai phân bậc 1' if use_diff else 'chuỗi gốc'}, n = {len(granger_data)}")

granger_results = []
try:
    gc = grangercausalitytests(granger_data, maxlag=MAX_GRANGER_LAG, verbose=False)
    for lag in range(1, MAX_GRANGER_LAG + 1):
        test_stats = gc[lag][0]
        # Lấy F-test (ssr_ftest)
        f_stat = test_stats["ssr_ftest"][0]
        f_pval = test_stats["ssr_ftest"][1]
        # Lấy chi2 test
        chi2_stat = test_stats["ssr_chi2test"][0]
        chi2_pval = test_stats["ssr_chi2test"][1]

        granger_results.append({
            "lag": lag,
            "F_stat": f_stat,
            "F_p_value": f_pval,
            "Chi2_stat": chi2_stat,
            "Chi2_p_value": chi2_pval,
            "sig_5%": "✓" if f_pval < 0.05 else "",
            "sig_10%": "✓" if f_pval < 0.10 else "",
        })
except Exception as e:
    print(f"  [LỖI] Granger test thất bại: {e}")

if granger_results:
    gc_df = pd.DataFrame(granger_results)
    print("\n>>> Bảng kết quả Granger Causality (GCF → GDP_growth)")
    print(gc_df.to_string(index=False, float_format="%.4f"))

    # Tìm lag có p-value nhỏ nhất
    best_gc = gc_df.loc[gc_df["F_p_value"].idxmin()]
    print(f"\n→ Lag có Granger causality mạnh nhất: lag = {int(best_gc['lag'])}")
    print(f"  F = {best_gc['F_stat']:.4f}, p = {best_gc['F_p_value']:.4f}")
    if best_gc['F_p_value'] < 0.05:
        print("  → Có ý nghĩa thống kê ở mức 5%: BÁC BỎ H0 → GCF Granger-cause GDP_growth")
    elif best_gc['F_p_value'] < 0.10:
        print("  → Có ý nghĩa thống kê ở mức 10%: BÁC BỎ H0 → GCF Granger-cause GDP_growth")
    else:
        print("  → KHÔNG có ý nghĩa thống kê → KHÔNG BÁC BỎ H0")

    # --- Biểu đồ Granger p-values ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_gc = ['#2ca02c' if p < 0.05 else ('#ff7f0e' if p < 0.10 else '#1f77b4')
                 for p in gc_df["F_p_value"]]
    ax.bar(gc_df["lag"], gc_df["F_stat"], color=colors_gc, edgecolor="black", width=0.5)
    ax.set_xlabel("Lag (kỳ trễ)")
    ax.set_ylabel("F-statistic")
    ax.set_title("Granger Causality: GCF → GDP Growth")
    ax.set_xticks(gc_df["lag"].astype(int).tolist())

    # Thêm p-value trên mỗi cột
    for _, row in gc_df.iterrows():
        ax.text(row["lag"], row["F_stat"] + 0.05, f'p={row["F_p_value"]:.3f}',
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.5)
    legend_elements_gc = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='p < 0.05 (sig 5%)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='p < 0.10 (sig 10%)'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='p ≥ 0.10 (not sig)'),
    ]
    ax.legend(handles=legend_elements_gc, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_granger_causality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Lưu biểu đồ] {OUTPUT_DIR / '04_granger_causality.png'}")

# ====================================================================
# TỔNG HỢP KẾT LUẬN
# ====================================================================
print("\n" + "=" * 70)
print("TỔNG HỢP KẾT LUẬN")
print("=" * 70)

print(f"""
1. CROSS-CORRELATION (Investment(t-k) vs ΔGDP(t)):
   → Lag k={int(best_cc['lag_k'])} cho hệ số tương quan cao nhất
     (r = {best_cc['pearson_r']:.4f}, p = {best_cc['p_value']:.4f})

2. ICOR theo Lag:
   → Lag k={int(best_icor_lag['lag_k'])} cho ICOR ổn định & hợp lý nhất
     (Mean = {best_icor_lag['mean']:.2f}, CV = {best_icor_lag['CV%']:.1f}%,
      {best_icor_lag['%_in_3-6']:.1f}% nằm trong khoảng [3–6])
""")

if granger_results:
    print(f"""3. GRANGER CAUSALITY (GCF → GDP_growth):
   → Lag = {int(best_gc['lag'])} có Granger causality mạnh nhất
     (F = {best_gc['F_stat']:.4f}, p = {best_gc['F_p_value']:.4f})
""")

print("Giải thích kinh tế:")
print("  - Với nước đang phát triển như Việt Nam, ICOR hợp lý thường rơi")
print("    vào khoảng 3–6. Lag k phù hợp giúp phản ánh đúng thời gian")
print("    để đầu tư phát huy hiệu quả vào tăng trưởng GDP.")
print("  - Lag càng lớn → đầu tư cần thời gian dài hơn để tác động.")
print("  - Lag k=0 giả định đầu tư tác động ngay lập tức (ít thực tế).")
print("  - Lag k=1-2 thường hợp lý cho Việt Nam (giai đoạn xây dựng,")
print("    triển khai dự án mất 1–2 năm).")

print("\n" + "=" * 70)
print("PHÂN TÍCH HOÀN TẤT. Kết quả lưu tại:", OUTPUT_DIR)
print("=" * 70)
