import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ========== CONTOH 1: Data Prediksi vs Aktual ==========
print("=" * 50)
print("CONTOH 1: Menghitung RMSE")
print("=" * 50)

# Data aktual dan prediksi
data_aktual = [100, 120, 130, 140, 150, 160, 170, 180, 190, 200]
data_prediksi = [98, 125, 128, 145, 148, 162, 168, 182, 188, 205]

# Cara 1: Manual
errors = []
for i in range(len(data_aktual)):
    error = data_aktual[i] - data_prediksi[i]
    errors.append(error)
    print(f"Data ke-{i+1}: Aktual={data_aktual[i]}, Prediksi={data_prediksi[i]}, Error={error}")

# Hitung RMSE manual
squared_errors = [e**2 for e in errors]
mean_squared_error_manual = sum(squared_errors) / len(squared_errors)
rmse_manual = mean_squared_error_manual ** 0.5

print(f"\nRMSE (Manual): {rmse_manual:.4f}")

# Cara 2: Menggunakan NumPy
rmse_numpy = np.sqrt(np.mean((np.array(data_aktual) - np.array(data_prediksi))**2))
print(f"RMSE (NumPy): {rmse_numpy:.4f}")

# Cara 3: Menggunakan sklearn
rmse_sklearn = np.sqrt(mean_squared_error(data_aktual, data_prediksi))
print(f"RMSE (sklearn): {rmse_sklearn:.4f}")

# ========== CONTOH 2: Menghitung Standard Deviation ==========
print("\n" + "=" * 50)
print("CONTOH 2: Menghitung Standard Deviation")
print("=" * 50)

# Data nilai siswa
nilai_siswa = [75, 80, 85, 70, 90, 88, 92, 78, 83, 87, 95, 72, 89, 91, 84]

print(f"Data nilai siswa: {nilai_siswa}")
print(f"Jumlah data: {len(nilai_siswa)}")

# Cara 1: Manual
mean = sum(nilai_siswa) / len(nilai_siswa)
print(f"\nMean (rata-rata): {mean:.2f}")

variance = sum((x - mean)**2 for x in nilai_siswa) / len(nilai_siswa)
std_dev_population = variance ** 0.5

print(f"Variance (populasi): {variance:.2f}")
print(f"Standard Deviation (populasi): {std_dev_population:.2f}")

# Standard deviation sampel (n-1)
variance_sample = sum((x - mean)**2 for x in nilai_siswa) / (len(nilai_siswa) - 1)
std_dev_sample = variance_sample ** 0.5
print(f"Standard Deviation (sampel): {std_dev_sample:.2f}")

# Cara 2: Menggunakan NumPy
std_numpy_pop = np.std(nilai_siswa)  # populasi (ddof=0)
std_numpy_sample = np.std(nilai_siswa, ddof=1)  # sampel (ddof=1)

print(f"\nStandard Deviation NumPy (populasi): {std_numpy_pop:.2f}")
print(f"Standard Deviation NumPy (sampel): {std_numpy_sample:.2f}")

# ========== CONTOH 3: Analisis Lengkap dengan Pandas ==========
print("\n" + "=" * 50)
print("CONTOH 3: Analisis Lengkap dengan Pandas")
print("=" * 50)

# Buat dataset
data = {
    'Bulan': ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Ags', 'Sep', 'Okt'],
    'Penjualan_Aktual': [120, 135, 150, 145, 160, 175, 170, 185, 190, 200],
    'Penjualan_Prediksi': [118, 140, 148, 150, 158, 178, 168, 188, 192, 198]
}

df = pd.DataFrame(data)
df['Error'] = df['Penjualan_Aktual'] - df['Penjualan_Prediksi']
df['Squared_Error'] = df['Error'] ** 2

print(df)

# Hitung metrik
rmse = np.sqrt(df['Squared_Error'].mean())
mae = df['Error'].abs().mean()
std_error = df['Error'].std()

print(f"\nMetrik Evaluasi:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"Standard Deviation dari Error: {std_error:.4f}")

# Statistik deskriptif
print(f"\nStatistik Penjualan Aktual:")
print(f"Mean: {df['Penjualan_Aktual'].mean():.2f}")
print(f"Std Dev: {df['Penjualan_Aktual'].std():.2f}")
print(f"Min: {df['Penjualan_Aktual'].min()}")
print(f"Max: {df['Penjualan_Aktual'].max()}")

# ========== CONTOH 4: Visualisasi ==========
print("\n" + "=" * 50)
print("CONTOH 4: Membuat Visualisasi")
print("=" * 50)

plt.figure(figsize=(12, 5))
# ========== CONTOH 4: Visualisasi ==========
print("\n" + "=" * 50)
print("CONTOH 4: Membuat Visualisasi")
print("=" * 50)

plt.figure(figsize=(12, 5))

# Plot 1: Perbandingan Aktual vs Prediksi
plt.subplot(1, 2, 1)
plt.plot(df['Bulan'], df['Penjualan_Aktual'], marker='o', label='Aktual', linewidth=2)
plt.plot(df['Bulan'], df['Penjualan_Prediksi'], marker='s', label='Prediksi', linewidth=2)
plt.xlabel('Bulan')
plt.ylabel('Penjualan')
plt.title('Perbandingan Penjualan Aktual vs Prediksi')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Distribusi Error
plt.subplot(1, 2, 2)
plt.bar(df['Bulan'], df['Error'], color=['red' if x < 0 else 'green' for x in df['Error']])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Bulan')
plt.ylabel('Error')
plt.title('Distribusi Error Prediksi')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Simpan gambar
plt.savefig('analisis_rmse_sd.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'analisis_rmse_sd.png'")

# TAMBAHKAN BARIS INI AGAR VISUALISASI MUNCUL
plt.show()

# Plot 1: Perbandingan Aktual vs Prediksi
plt.subplot(1, 2, 1)
plt.plot(df['Bulan'], df['Penjualan_Aktual'], marker='o', label='Aktual', linewidth=2)
plt.plot(df['Bulan'], df['Penjualan_Prediksi'], marker='s', label='Prediksi', linewidth=2)
plt.xlabel('Bulan')
plt.ylabel('Penjualan')
plt.title('Perbandingan Penjualan Aktual vs Prediksi')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Distribusi Error
plt.subplot(1, 2, 2)
plt.bar(df['Bulan'], df['Error'], color=['red' if x < 0 else 'green' for x in df['Error']])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Bulan')
plt.ylabel('Error')
plt.title('Distribusi Error Prediksi')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_rmse_sd.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'analisis_rmse_sd.png'")

# ========== CONTOH 5: Fungsi Reusable ==========
print("\n" + "=" * 50)
print("CONTOH 5: Fungsi yang Bisa Digunakan Ulang")
print("=" * 50)

def hitung_metrik(actual, predicted):
    """
    Menghitung berbagai metrik evaluasi
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Error
    errors = actual - predicted
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAE
    mae = np.mean(np.abs(errors))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(errors / actual)) * 100
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Standard Deviation
    std_actual = np.std(actual, ddof=1)
    std_predicted = np.std(predicted, ddof=1)
    std_errors = np.std(errors, ddof=1)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R-squared': r_squared,
        'Std_Actual': std_actual,
        'Std_Predicted': std_predicted,
        'Std_Errors': std_errors
    }

# Test fungsi
hasil = hitung_metrik(df['Penjualan_Aktual'], df['Penjualan_Prediksi'])

print("Hasil Perhitungan Metrik:")
for key, value in hasil.items():
    print(f"{key}: {value:.4f}")

print("\n" + "=" * 50)
print("SELESAI")
print("=" * 50)