"""Custom exceptions untuk Aplikasi Analisis Data."""


class AppError(Exception):
    """Kelas dasar untuk semua error aplikasi."""
    pass


class DataLoadError(AppError):
    """Dimunculkan saat data gagal dimuat."""
    pass


class KolomTidakDitemukanError(AppError):
    """Dimunculkan saat kolom yang diminta tidak ada."""
    pass


class ModelError(AppError):
    """Dimunculkan saat terjadi masalah pelatihan model."""
    pass
