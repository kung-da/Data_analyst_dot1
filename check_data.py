import pandas as pd
import os

BASE_DIR = r'd:\HK2- 2025-2026\DATA Analyst\Đợt 1'
file_map = {
    'gcf': os.path.join(BASE_DIR, 'GCF', 'API_NE.GDI.TOTL.ZS_DS2_en_csv_v2_115698.csv'),
    'gdp_growth': os.path.join(BASE_DIR, 'GDP', 'API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_121708.csv')
}
targets = ['Viet Nam', 'Korea, Rep.', 'China', 'Thailand', 'Indonesia']

for k, p in file_map.items():
    df = pd.read_csv(p, skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit() and 1990 <= int(c) <= 2023]
    print(f'\nIndicator: {k}')
    for t in targets:
        c_df = df[df['Country Name'] == t][year_cols]
        if not c_df.empty:
            non_null = c_df.notnull().sum().sum()
            total = len(year_cols)
            print(f'  {t}: {non_null}/{total}')
        else:
            print(f'  {t}: NOT FOUND')
