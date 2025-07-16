# Dockerfile
FROM python:3.10.13-slim

# 1) Install OS dependencies
RUN apt-get update && \
    apt-get install -y \
      python3-distutils \
      libgl1 \
      libglib2.0-0 \
      libgeos-dev \
      python3-shapely && \
    rm -rf /var/lib/apt/lists/*

# 2) Set working directory
WORKDIR /app

# 3) Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools==68.2.2 wheel && \
    pip install -r requirements.txt

# 4) Copy app code
COPY . .

# 5) Expose port and declare entrypoint
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
