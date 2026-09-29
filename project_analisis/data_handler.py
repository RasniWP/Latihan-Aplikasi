"""Modul pemuatan & pembersihan data."""
import pandas as pd
from exceptions import DataLoadError, KolomTidakDitemukanError


class DataLoader:
    """Bertanggung jawab memuat dataset dari berbagai sumber."""

    @staticmethod
    def dari_csv(path):
        """Muat CSV menjadi DataFrame. Lempar DataLoadError jika gagal."""
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise DataLoadError('Dataset kosong.')
            return df
        except FileNotFoundError:
            raise DataLoadError(f'File tidak ditemukan: {path}')
        except pd.errors.EmptyDataError:
            raise DataLoadError('File tidak berisi data.')


class DataCleaner:
    """Membersihkan DataFrame: missing values, duplikat, tipe data."""

    def __init__(self, df):
        self.df = df.copy()          # bekerja di salinan (data asli aman)
        self._log = []               # catatan langkah pembersihan

    def tangani_missing(self, strategi='median'):
        """Isi missing value: numerik pakai median/mean, kategorik pakai modus."""
        for kol in self.df.columns:
            if self.df[kol].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(self.df[kol]):
                nilai = (self.df[kol].median() if strategi == 'median'
                         else self.df[kol].mean())
            else:
                modus = self.df[kol].mode()
                nilai = modus[0] if not modus.empty else 'Tidak Diketahui'
            self.df[kol] = self.df[kol].fillna(nilai)
            self._log.append(f'Kolom "{kol}" diisi dengan {nilai}')
        return self

    def hapus_duplikat(self):
        sebelum = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dihapus = sebelum - len(self.df)
        if dihapus:
            self._log.append(f'{dihapus} baris duplikat dihapus')
        return self

    def hasil(self):
        """Kembalikan DataFrame yang sudah bersih."""
        return self.df

    def laporan(self):
        """Kembalikan log langkah pembersihan."""
        return self._log if self._log else ['Tidak ada pembersihan yang diperlukan.']
