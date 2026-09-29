"""
Dashboard Analisis & Prediksi Data
Aplikasi web interaktif berbasis Streamlit yang memakai AppEngine (Modul 6).
Mentor: Mega Bagus Herlambang
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Pastikan package engine (Modul 6) bisa ditemukan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project_analisis'))
from engine import AppEngine
from exceptions import AppError

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title='Dashboard Analisis Data',
    page_icon='📊',
    layout='wide',
)

st.title('📊 Dashboard Analisis & Prediksi Data')
st.caption('Project akhir kurikulum Python - Mentor: Mega Bagus Herlambang')

# ------------------------------------------------------------------
# Inisialisasi session_state (menyimpan engine antar-interaksi)
# ------------------------------------------------------------------
if 'engine' not in st.session_state:
    st.session_state.engine = AppEngine()
if 'data_dimuat' not in st.session_state:
    st.session_state.data_dimuat = False
if 'sudah_bersih' not in st.session_state:
    st.session_state.sudah_bersih = False

engine = st.session_state.engine

# ------------------------------------------------------------------
# SIDEBAR: unggah file & navigasi
# ------------------------------------------------------------------
with st.sidebar:
    st.header('⚙️ Panel Kontrol')
    file = st.file_uploader('Unggah file CSV Anda', type=['csv'])

    if file is not None:
        try:
            # simpan sementara lalu muat lewat engine
            df_tmp = pd.read_csv(file)
            engine.df_asli = df_tmp
            engine.df_bersih = df_tmp.copy()
            st.session_state.data_dimuat = True
            st.success(f'Data dimuat: {df_tmp.shape[0]} baris, {df_tmp.shape[1]} kolom')
        except Exception as e:
            st.error(f'Gagal memuat file: {e}')
            st.session_state.data_dimuat = False

    st.divider()
    menu = st.radio(
        'Navigasi',
        ['🏠 Beranda', '🔍 Eksplorasi Data', '📈 Visualisasi', '🤖 Machine Learning'],
    )


# ==================================================================
# Fungsi bantu: tampilkan pesan bila belum ada data
# ==================================================================
def butuh_data():
    if not st.session_state.data_dimuat:
        st.info('👈 Silakan unggah file CSV terlebih dahulu melalui panel di kiri.')
        return True
    return False


# ==================================================================
# HALAMAN: BERANDA
# ==================================================================
if menu == '🏠 Beranda':
    st.header('Selamat Datang')
    st.write(
        'Aplikasi ini membantu Anda menganalisis dataset apa pun dan '
        'membangun model prediksi - cukup dengan beberapa klik.'
    )

    if butuh_data():
        st.stop()

    analyzer = engine.analisis()
    r = analyzer.ringkasan()

    # Tampilkan KPI dalam beberapa kolom
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Jumlah Baris', r['jumlah_baris'])
    k2.metric('Jumlah Kolom', r['jumlah_kolom'])
    k3.metric('Kolom Numerik', len(r['kolom_numerik']))
    k4.metric('Nilai Hilang', r['total_missing'])

    st.subheader('💡 Insight Otomatis')
    for ins in analyzer.insight_otomatis():
        st.write('- ' + ins)

    st.subheader('Cuplikan Data')
    st.dataframe(engine.df_bersih.head(10), use_container_width=True)


# ==================================================================
# HALAMAN: EKSPLORASI DATA
# ==================================================================
elif menu == '🔍 Eksplorasi Data':
    st.header('🔍 Eksplorasi Data')
    if butuh_data():
        st.stop()

    st.subheader('Tabel Data')
    st.dataframe(engine.df_bersih, use_container_width=True)

    st.subheader('Statistik Deskriptif')
    analyzer = engine.analisis()
    st.dataframe(analyzer.statistik(), use_container_width=True)

    st.subheader('Informasi Tipe Data & Nilai Hilang')
    info = pd.DataFrame({
        'Tipe Data': engine.df_bersih.dtypes.astype(str),
        'Nilai Hilang': engine.df_bersih.isnull().sum(),
        'Nilai Unik': engine.df_bersih.nunique(),
    })
    st.dataframe(info, use_container_width=True)

    st.divider()
    st.subheader('🧹 Pembersihan Data')
    strategi = st.selectbox(
        'Strategi pengisian nilai numerik yang hilang',
        ['median', 'mean'],
    )
    if st.button('Bersihkan Data Sekarang'):
        try:
            _, laporan = engine.bersihkan(strategi=strategi)
            st.session_state.sudah_bersih = True
            st.success('Data berhasil dibersihkan!')
            st.write('**Laporan pembersihan:**')
            for baris in laporan:
                st.write('- ' + baris)
        except AppError as e:
            st.error(f'Kesalahan: {e}')


# ==================================================================
# HALAMAN: VISUALISASI
# ==================================================================
elif menu == '📈 Visualisasi':
    st.header('📈 Visualisasi Data')
    if butuh_data():
        st.stop()

    df = engine.df_bersih
    kol_numerik = df.select_dtypes('number').columns.tolist()
    kol_kategorik = df.select_dtypes('object').columns.tolist()

    jenis = st.selectbox(
        'Pilih jenis visualisasi',
        ['Histogram', 'Scatter Plot', 'Bar Kategori', 'Heatmap Korelasi'],
    )

    sns.set_style('whitegrid')

    if jenis == 'Histogram':
        kol = st.selectbox('Pilih kolom numerik', kol_numerik)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[kol], kde=True, ax=ax, color='#4C72B0')
        ax.set_title(f'Distribusi {kol}')
        st.pyplot(fig)

    elif jenis == 'Scatter Plot':
        c1, c2 = st.columns(2)
        x = c1.selectbox('Sumbu X', kol_numerik, index=0)
        y = c2.selectbox('Sumbu Y', kol_numerik,
                         index=min(1, len(kol_numerik) - 1))
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df, x=x, y=y, ax=ax, alpha=0.6)
        ax.set_title(f'{x} vs {y}')
        st.pyplot(fig)

    elif jenis == 'Bar Kategori':
        if not kol_kategorik:
            st.warning('Tidak ada kolom kategorik pada dataset ini.')
        else:
            kol = st.selectbox('Pilih kolom kategorik', kol_kategorik)
            hitung = df[kol].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=hitung.values, y=hitung.index, ax=ax, palette='viridis')
            ax.set_title(f'Frekuensi {kol}')
            ax.set_xlabel('Jumlah')
            st.pyplot(fig)

    elif jenis == 'Heatmap Korelasi':
        korr = engine.analisis().korelasi()
        if korr is None:
            st.warning('Perlu minimal 2 kolom numerik untuk heatmap korelasi.')
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(korr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title('Matriks Korelasi')
            st.pyplot(fig)


# ==================================================================
# HALAMAN: MACHINE LEARNING
# ==================================================================
elif menu == '🤖 Machine Learning':
    st.header('🤖 Latih Model Prediksi')
    if butuh_data():
        st.stop()

    df = engine.df_bersih
    semua_kolom = df.columns.tolist()

    st.write('Pilih apa yang ingin diprediksi dan fitur apa yang dipakai.')
    target = st.selectbox('🎯 Kolom Target (yang ingin diprediksi)', semua_kolom)
    fitur = st.multiselect(
        '📋 Kolom Fitur (input model)',
        [k for k in semua_kolom if k != target],
        default=[k for k in semua_kolom if k != target][:4],
    )

    c1, c2 = st.columns(2)
    jenis = c1.selectbox('Jenis tugas', ['auto', 'regresi', 'klasifikasi'])
    algoritma = c2.selectbox('Algoritma', ['forest', 'linear'])

    if st.button('🚀 Latih Model'):
        if not fitur:
            st.warning('Pilih minimal satu kolom fitur.')
        else:
            try:
                with st.spinner('Melatih model...'):
                    trainer = engine.buat_trainer()
                    metrik = trainer.latih(
                        fitur=fitur, target=target,
                        jenis=jenis, algoritma=algoritma,
                    )
                st.success(f'Model selesai dilatih! (Jenis tugas: {trainer.jenis})')

                st.subheader('📊 Hasil Evaluasi')
                kolom_metrik = st.columns(len(metrik))
                for kol, (nama, nilai) in zip(kolom_metrik, metrik.items()):
                    kol.metric(nama, nilai)

                if trainer.jenis == 'regresi':
                    st.caption('R2 mendekati 1 = model sangat baik. MAE = rata-rata kesalahan prediksi.')
                else:
                    st.caption('Akurasi = proporsi prediksi yang benar (0 - 1).')
            except AppError as e:
                st.error(f'Kesalahan: {e}')
            except Exception as e:
                st.error(f'Terjadi kesalahan tak terduga: {e}')
