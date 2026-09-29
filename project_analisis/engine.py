"""Engine utama aplikasi - menyatukan semua subsistem (Facade Pattern)."""
from data_handler import DataLoader, DataCleaner
from analyzer import DataAnalyzer
from model_trainer import ModelTrainer
from exceptions import AppError


class AppEngine:
    """Titik masuk tunggal untuk seluruh fungsi aplikasi."""

    def __init__(self):
        self.df_asli = None
        self.df_bersih = None

    def muat_csv(self, path):
        self.df_asli = DataLoader.dari_csv(path)
        self.df_bersih = self.df_asli.copy()
        return self.df_asli

    def bersihkan(self, strategi='median'):
        cleaner = DataCleaner(self.df_asli)
        self.df_bersih = (cleaner
                          .tangani_missing(strategi)
                          .hapus_duplikat()
                          .hasil())
        return self.df_bersih, cleaner.laporan()

    def analisis(self):
        return DataAnalyzer(self.df_bersih)

    def buat_trainer(self):
        return ModelTrainer(self.df_bersih)
