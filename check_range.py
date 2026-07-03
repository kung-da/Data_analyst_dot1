import pandas as pd
import os

BASE_DIR = r'd:\HK2- 2025-2026\DATA Analyst\Đợt 1'
path = os.path.join(BASE_DIR, 'GCF', 'API_NE.GDI.TOTL.ZS_DS2_en_csv_v2_115698.csv')
df = pd.read_csv(path, skiprows=4)
targets = ['Viet Nam', 'Korea, Rep.', 'China', 'Thailand', 'Indonesia']

for t in targets:
    c_df = df[df['Country Name'] == t]
    valid_years = [c for c in c_df.columns if c.isdigit() and c_df[c].notnull().any()]
    if valid_years:
        print(f'{t}: {min(valid_years)} - {max(valid_years)} (count: {len([y for y in valid_years if 1990 <= int(y) <= 2023])}/34)')
    else:
        print(f'{t}: No data')
