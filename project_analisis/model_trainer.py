"""Modul pelatihan & evaluasi model machine learning."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, r2_score,
                             accuracy_score, classification_report)
from exceptions import ModelError, KolomTidakDitemukanError


class ModelTrainer:
    """Melatih model regresi/klasifikasi pada dataset bersih."""

    def __init__(self, df):
        self.df = df
        self.model = None
        self.fitur = None
        self.target = None
        self.jenis = None
        self.metrik = {}

    def _siapkan(self, fitur, target):
        for kol in fitur + [target]:
            if kol not in self.df.columns:
                raise KolomTidakDitemukanError(f'Kolom "{kol}" tidak ada.')
        data = self.df[fitur + [target]].copy()
        # encoding sederhana untuk kolom kategorik di fitur
        kol_kategorik = [c for c in fitur
                         if not pd.api.types.is_numeric_dtype(data[c])]
        if kol_kategorik:
            data = pd.get_dummies(data, columns=kol_kategorik, dtype=int)
        data = data.fillna(data.median(numeric_only=True))
        y = data[target]
        X = data.drop(columns=[target])
        return X, y

    def latih(self, fitur, target, jenis='auto', algoritma='linear'):
        """Latih model. jenis: regresi/klasifikasi/auto."""
        self.fitur, self.target = fitur, target
        X, y = self._siapkan(fitur, target)

        # Deteksi otomatis jenis tugas
        if jenis == 'auto':
            jenis = 'klasifikasi' if y.nunique() <= 10 and y.dtype != 'float64' else 'regresi'
        self.jenis = jenis

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Pilih model (polymorphism + abstraction)
        if jenis == 'regresi':
            self.model = (RandomForestRegressor(n_estimators=100, random_state=42)
                          if algoritma == 'forest' else LinearRegression())
        elif jenis == 'klasifikasi':
            self.model = (RandomForestClassifier(n_estimators=100, random_state=42)
                          if algoritma == 'forest'
                          else LogisticRegression(max_iter=1000))
        else:
            raise ModelError(f'Jenis tugas tidak dikenal: {jenis}')

        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)

        # Evaluasi sesuai jenis
        if jenis == 'regresi':
            self.metrik = {
                'MAE': round(mean_absolute_error(y_te, y_pred), 2),
                'R2': round(r2_score(y_te, y_pred), 3),
            }
        else:
            self.metrik = {
                'Akurasi': round(accuracy_score(y_te, y_pred), 3),
            }
        return self.metrik

    def prediksi(self, data_baru):
        """Prediksi pada data baru (list of dict atau DataFrame)."""
        if self.model is None:
            raise ModelError('Model belum dilatih.')
        return self.model.predict(data_baru)
