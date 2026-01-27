import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Membuat data penjualan contoh
np.random.seed(42)
n_records = 1000

# Generate data
dates = pd.date_range(start='2024-01-01', end='2024-12-31', periods=n_records)
products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headset']
regions = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar']

data = {
    'tanggal': np.random.choice(dates, n_records),
    'produk': np.random.choice(products, n_records),
    'region': np.random.choice(regions, n_records),
    'jumlah': np.random.randint(1, 100, n_records),
    'harga': np.random.choice([50000, 150000, 300000, 2000000, 8000000], n_records)
}

df = pd.DataFrame(data)
df['total_penjualan'] = df['jumlah'] * df['harga']
df = df.sort_values('tanggal').reset_index(drop=True)

print("=" * 70)
print("ANALISIS DATA PENJUALAN")
print("=" * 70)

# 1. Informasi dasar dataset
print("\n1. INFORMASI DATASET")
print("-" * 70)
print(f"Total Records: {len(df)}")
print(f"Periode: {df['tanggal'].min().date()} sampai {df['tanggal'].max().date()}")
print("\nContoh Data (5 baris pertama):")
print(df.head())

# 2. Statistik deskriptif
print("\n\n2. STATISTIK DESKRIPTIF")
print("-" * 70)
print(df[['jumlah', 'harga', 'total_penjualan']].describe())

# 3. Total penjualan per produk
print("\n\n3. TOTAL PENJUALAN PER PRODUK")
print("-" * 70)
penjualan_produk = df.groupby('produk').agg({
    'total_penjualan': 'sum',
    'jumlah': 'sum'
}).sort_values('total_penjualan', ascending=False)
penjualan_produk['total_penjualan'] = penjualan_produk['total_penjualan'].apply(lambda x: f"Rp {x:,.0f}")
print(penjualan_produk)

# 4. Total penjualan per region
print("\n\n4. TOTAL PENJUALAN PER REGION")
print("-" * 70)
penjualan_region = df.groupby('region')['total_penjualan'].sum().sort_values(ascending=False)
penjualan_region = penjualan_region.apply(lambda x: f"Rp {x:,.0f}")
print(penjualan_region)

# 5. Penjualan bulanan
print("\n\n5. TREN PENJUALAN BULANAN")
print("-" * 70)
df['bulan'] = df['tanggal'].dt.to_period('M')
penjualan_bulanan = df.groupby('bulan')['total_penjualan'].sum()
print(penjualan_bulanan.apply(lambda x: f"Rp {x:,.0f}"))

# 6. Produk terlaris
print("\n\n6. PRODUK TERLARIS (Berdasarkan Kuantitas)")
print("-" * 70)
produk_terlaris = df.groupby('produk')['jumlah'].sum().sort_values(ascending=False)
print(produk_terlaris)

# 7. Analisis kombinasi produk dan region
print("\n\n7. TOP 5 KOMBINASI PRODUK-REGION")
print("-" * 70)
kombinasi = df.groupby(['produk', 'region'])['total_penjualan'].sum().sort_values(ascending=False).head()
print(kombinasi.apply(lambda x: f"Rp {x:,.0f}"))

# 8. Metrik kinerja
print("\n\n8. METRIK KINERJA")
print("-" * 70)
total_revenue = df['total_penjualan'].sum()
avg_transaction = df['total_penjualan'].mean()
total_items = df['jumlah'].sum()

print(f"Total Revenue: Rp {total_revenue:,.0f}")
print(f"Rata-rata Transaksi: Rp {avg_transaction:,.0f}")
print(f"Total Item Terjual: {total_items:,}")
print(f"Rata-rata Item per Transaksi: {df['jumlah'].mean():.2f}")

# 9. Pivot table
print("\n\n9. PIVOT TABLE: PENJUALAN PRODUK PER REGION")
print("-" * 70)
pivot = pd.pivot_table(df, 
                       values='total_penjualan', 
                       index='produk', 
                       columns='region', 
                       aggfunc='sum', 
                       fill_value=0)
print(pivot.applymap(lambda x: f"{x/1000000:.1f}M" if x > 0 else "-"))

print("\n" + "=" * 70)
print("ANALISIS SELESAI")
print("=" * 70)