"""Modul analisis & ringkasan data (versi final)."""
import numpy as np
import pandas as pd
from exceptions import KolomTidakDitemukanError


class DataAnalyzer:
    """Menghasilkan statistik deskriptif & insight dari DataFrame."""

    def __init__(self, df):
        self.df = df

    def ringkasan(self):
        return {
            'jumlah_baris': len(self.df),
            'jumlah_kolom': self.df.shape[1],
            'kolom_numerik': self.df.select_dtypes('number').columns.tolist(),
            'kolom_kategorik': self.df.select_dtypes('object').columns.tolist(),
            'total_missing': int(self.df.isnull().sum().sum()),
            'memori_kb': round(self.df.memory_usage(deep=True).sum() / 1024, 1),
        }

    def statistik(self):
        return self.df.describe()

    def korelasi(self):
        num = self.df.select_dtypes('number')
        if num.shape[1] < 2:
            return None
        return num.corr()

    def distribusi_kategori(self, kolom):
        if kolom not in self.df.columns:
            raise KolomTidakDitemukanError(f'Kolom "{kolom}" tidak ada.')
        return self.df[kolom].value_counts()

    def insight_otomatis(self):
        out = []
        r = self.ringkasan()
        out.append(f"Dataset memiliki {r['jumlah_baris']} baris dan {r['jumlah_kolom']} kolom "
                   f"({len(r['kolom_numerik'])} numerik, {len(r['kolom_kategorik'])} kategorik).")
        if r['total_missing'] > 0:
            out.append(f"Terdapat {r['total_missing']} nilai hilang yang perlu ditangani.")
        else:
            out.append('Tidak ada nilai hilang - data sudah lengkap.')
        korr = self.korelasi()
        if korr is not None and len(korr) >= 2:
            mask = ~np.eye(len(korr), dtype=bool)
            abs_korr = korr.abs().where(mask)
            maks = abs_korr.stack().idxmax()
            nilai = korr.loc[maks[0], maks[1]]
            out.append(f"Korelasi terkuat: '{maks[0]}' & '{maks[1]}' (r = {nilai:.2f}).")
        return out
