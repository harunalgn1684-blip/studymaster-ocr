# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies (Tesseract OCR and GL libraries for OpenCV)
# Updated libgl1-mesa-glx to libgl1 for newer Debian versions
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-tur \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Gunicorn is needed for production server
RUN pip install gunicorn

# Copy the rest of the application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port 5000 (though Render handles this via env var usually, good for doc)
EXPOSE 5000

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]
