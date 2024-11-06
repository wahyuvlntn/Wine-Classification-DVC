
# Wine Classification Project with DVC

Proyek ini menunjukkan workflow machine learning sederhana menggunakan DVC untuk versioning dataset dan model. Dataset yang digunakan adalah *Wine* dari `sklearn.datasets`, dan model yang dibangun menggunakan logistic regression.

## Struktur Proyek

- **data/**: Menyimpan dataset yang sudah di-preprocess (`wine_train.csv` dan `wine_test.csv`)
- **src/**: 
  - `preprocess.py`: Script untuk memproses data.
  - `train.py`: Script untuk melatih model.
- **model/**: Menyimpan file model yang sudah dilatih (`model.pkl`)
- **dvc.yaml**: File DVC pipeline untuk mengatur workflow
- **params.yaml**: Parameter yang dapat diubah untuk model
- **dvc.lock**: Lock file dari DVC yang melacak versi dataset dan model
- **README.md**: Dokumentasi proyek

## Persyaratan

- Python 3.x
- Virtual environment (`venv`)
- DVC
- Git
- Dependencies (`requirements.txt`)

## Langkah Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/wahyuvlntn/Wine-Classification-DVC
   cd wine-classification
   ```

2. **Buat virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Aktivasi di Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install dvc scikit-learn pandas joblib
   ```

4. **Inisialisasi DVC dan Git** (jika belum dilakukan):
   ```bash
   dvc init
   git init
   git add .
   git commit -m "Initial commit with DVC and Git"
   ```

## Menjalankan Pipeline

- Untuk menjalankan pipeline (preprocessing dan training model), gunakan:
  ```bash
  dvc repro
  ```

- Pipeline ini akan:
  - Preprocess dataset dan menyimpan `wine_train.csv` dan `wine_test.csv` dalam folder `data/`.
  - Melatih model logistic regression dan menyimpan model sebagai `model/model.pkl`.

## Versioning Dataset dan Model

- **Dataset**: DVC melacak perubahan pada dataset `wine_train.csv` dan `wine_test.csv`.
- **Model**: Model `model.pkl` di-track sebagai output dari tahap training.

## Menggunakan Versi Dataset atau Model Sebelumnya

- **Lihat Riwayat Versi**: 
  ```bash
  dvc diff
  ```

- **Kembalikan ke Versi Sebelumnya**:
  ```bash
  git checkout <commit_id>
  dvc checkout
  ```

## Menyimpan ke Remote (Opsional)

1. **Tambah Remote Storage**:
   ```bash
   dvc remote add -d myremote /path/to/remote
   ```

2. **Push ke Remote**:
   ```bash
   dvc push
   ```

