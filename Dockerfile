# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port for Uvicorn
EXPOSE 10000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
