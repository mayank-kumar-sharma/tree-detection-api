# Dockerfile
FROM python:3.10.13-slim

# 1️⃣ Install OS dependencies
RUN apt-get update && \
    apt-get install -y \
      python3-distutils \
      libgl1 \
      libglib2.0-0 \
      libgeos-dev \
      python3-shapely && \
    rm -rf /var/lib/apt/lists/*

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and install Python dependencies (no cache to slim image)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools==68.2.2 wheel && \
    pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy application code into the container
COPY . .

# 5️⃣ Expose the port and specify the startup command
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

