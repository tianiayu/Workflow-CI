# Gunakan image Python ringan
FROM python:3.12-slim

# Buat working directory
WORKDIR /app

# Copy requirements dan install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Salin file model dan skrip inference
COPY model.joblib .
COPY infer.py .

# Jalankan skrip prediksi saat container start
CMD ["python", "infer.py"]
